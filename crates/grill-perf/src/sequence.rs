use crate::model::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::rc::Rc;

#[cfg(test)]
#[path = "../tests/support/tool_assembly.rs"]
mod tool_assembly_tests;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Step {
    pub history: String,
    pub parent: Option<String>,
    pub cache: Cache,
    pub expect: Expected,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Fact {
    fact: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Expected {
    Json { value: Fact, strict: Option<String> },
    Tool { key: String, result: String },
}

/// Prospective fixed-call semantics, derived from the admitted step and its
/// actual retained parent history, never from provider output.
#[derive(Clone, Debug)]
pub(crate) struct ToolExpectation {
    pub key: String,
    pub prior_ids: Vec<String>,
    pub result: String,
    pub history_bytes: usize,
    pub history_messages: usize,
    pub history_cap: usize,
    pub message_cap: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ToolArguments {
    key: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ToolFunctionDelta {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ToolDelta {
    index: u32,
    id: Option<String>,
    #[serde(rename = "type")]
    kind: Option<String>,
    function: Option<ToolFunctionDelta>,
}

/// One fixed call; fragments remain data. No tool dispatch or execution exists.
pub(crate) struct ToolStream {
    expected: ToolExpectation,
    id: Option<String>,
    kind: bool,
    name: String,
    arguments: String,
    pub first_delta_us: Option<u64>,
    candidate_us: Option<u64>,
}

impl ToolStream {
    pub fn new(expected: ToolExpectation) -> Self {
        Self {
            expected,
            id: None,
            kind: false,
            name: String::new(),
            arguments: String::new(),
            first_delta_us: None,
            candidate_us: None,
        }
    }

    pub fn delta(&mut self, raw: &str, observed: u64) -> Result<()> {
        // A fixed-size array rejects additional calls before allocating them.
        let [delta]: [ToolDelta; 1] = serde_json::from_str(raw)
            .map_err(|_| "tool delta requires exactly one well-formed indexed call")?;
        if delta.index != 0 {
            return Err("fixed tool stream requires call index zero".into());
        }
        if let Some(id) = &delta.id
            && (!identifier(id) || self.id.is_some() || self.expected.prior_ids.contains(id))
        {
            return Err("invalid, duplicate or reused tool call ID".into());
        }
        if let Some(kind) = &delta.kind
            && (kind != "function" || self.kind)
        {
            return Err("invalid or duplicate tool call type".into());
        }
        let (name, arguments) = delta.function.as_ref().map_or(("", ""), |f| {
            (
                f.name.as_deref().unwrap_or(""),
                f.arguments.as_deref().unwrap_or(""),
            )
        });
        if self.name.len() + name.len() > "lookup_fact".len()
            || !("lookup_fact"[self.name.len()..]).starts_with(name)
            || self.arguments.len() + arguments.len() > 4096
        {
            return Err("tool name or argument fragments exceed the fixed call bounds".into());
        }
        // Commit only after the entire fragment is accepted.
        if let Some(id) = delta.id {
            self.id = Some(id);
        }
        self.kind |= delta.kind.is_some();
        self.name.push_str(name);
        self.arguments.push_str(arguments);
        if !name.is_empty() || !arguments.is_empty() {
            self.first_delta_us.get_or_insert(observed);
        }
        if self.valid() {
            self.candidate_us.get_or_insert(observed);
        }
        Ok(())
    }

    fn valid(&self) -> bool {
        self.id.is_some()
            && self.kind
            && self.name == "lookup_fact"
            && serde_json::from_str::<ToolArguments>(&self.arguments)
                .is_ok_and(|arguments| arguments.key == self.expected.key)
    }

    pub fn validated_us(&self) -> Result<u64> {
        if !self.valid() || self.first_delta_us.is_none() {
            return Err("incomplete or invalid fixed lookup_fact call".into());
        }
        let appended = json!([self.message_value(), {
            "role":"tool","tool_call_id":self.id,"content":self.expected.result
        }]);
        let appended_bytes = serde_json::to_vec(&appended)
            .map_err(|e| e.to_string())?
            .len();
        let retained_bytes = if self.expected.history_messages == 0 {
            appended_bytes
        } else {
            self.expected
                .history_bytes
                .checked_add(appended_bytes)
                .and_then(|bytes| bytes.checked_sub(1))
                .ok_or("tool retained history size overflow")?
        };
        if self.expected.history_messages > self.expected.message_cap.saturating_sub(2)
            || retained_bytes > self.expected.history_cap
        {
            return Err("fixed tool call exceeds retained history bounds".into());
        }
        self.candidate_us
            .ok_or_else(|| "missing valid tool call boundary".into())
    }

    pub fn message(&self) -> Result<Value> {
        self.validated_us()?;
        Ok(self.message_value())
    }

    fn message_value(&self) -> Value {
        json!({"role":"assistant","content":null,"tool_calls":[{
            "id":self.id,"type":"function","function":{
                "name":self.name,"arguments":self.arguments
            }
        }]})
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Check {
    pub history: String,
    pub parent: Option<String>,
    pub correct: bool,
    pub canonical_match: Option<bool>,
    pub strict_match: Option<bool>,
    pub error: Option<String>,
}

pub fn passed(check: &Check) -> bool {
    check.correct && check.strict_match != Some(false) && check.error.is_none()
}

pub fn validate(workload: &Workload) -> Result<()> {
    let sequence = crate::acquisition::conversation(workload);
    if !sequence {
        if workload.cases.iter().any(|case| case.step.is_some())
            || matches!(
                workload.request.profile,
                Profile::VllmConversationV2 | Profile::VllmConversationV3
            )
        {
            return Err(
                "sequence steps and conversation profiles require workload version 2 or 5".into(),
            );
        }
        return Ok(());
    }
    let profile = if matches!(workload.version, 5 | 6) {
        Profile::VllmConversationV3
    } else {
        Profile::VllmConversationV2
    };
    if workload.request.profile != profile
        || !workload.request.stream
        || workload.request.output.mode != OutputMode::Cap
        || workload.request.cache != Cache::Observe
        || workload.cases.len() > if workload.version == 6 { 128 } else { 16 }
        || workload.cells.len() != workload.cases.len()
        || workload.limits.response_bytes > 64 * 1024
    {
        return Err("conversation workload requires its exact versioned profile, streaming, capped output, per-step cache declarations and at most 16 ordered C1 steps".into());
    }
    for (index, (case, cell)) in workload.cases.iter().zip(&workload.cells).enumerate() {
        let step = case
            .step
            .as_ref()
            .ok_or("every conversation case requires a step")?;
        if !identifier(&step.history)
            || cell.case != case.id
            || cell.concurrency != 1
            || cell.trials != 1
            || cell.warmup_trials != 0
            || case
                .messages
                .iter()
                .any(|m| m.role != Role::User && m.role != Role::System)
            || case.messages.iter().any(|m| m.content.contains("{salt}"))
        {
            return Err("sequence requires ordered single-use C1 cases, declared history and user/system inputs without attempt salt".into());
        }
        if let Some(parent) = &step.parent {
            let prior = workload.cases[..index]
                .iter()
                .find(|prior| &prior.id == parent)
                .ok_or("sequence parent must reference an earlier step")?;
            if prior.step.as_ref().map(|step| &step.history) != Some(&step.history)
                || case.messages.iter().any(|m| m.role != Role::User)
                || step.cache == Cache::ReportedPrefixZero
                || case.fill.is_some()
            {
                return Err("sequence parent must belong to the same history; continuations append user messages only".into());
            }
        }
        if step.cache == Cache::ReportedPrefixHit
            && !workload.cases[..index].iter().any(|prior| {
                prior
                    .step
                    .as_ref()
                    .is_some_and(|s| s.history == step.history)
            })
        {
            return Err(
                "required prefix hit needs an earlier explicit prime in that history".into(),
            );
        }
        match &step.expect {
            Expected::Json { value, strict } => {
                if serde_json::to_vec(value).map_err(|e| e.to_string())?.len() > 4096 {
                    return Err("sequence expected JSON must be a bounded object".into());
                }
                if let Some(text) = strict
                    && (text.len() > 4096
                        || serde_json::from_str::<Fact>(text).ok().as_ref() != Some(value))
                {
                    return Err("strict expected text must encode the declared JSON object".into());
                }
            }
            Expected::Tool { key, result } => {
                if !identifier(key) || result.len() > 4096 {
                    return Err("fixed lookup_fact fixture exceeds key/result bounds".into());
                }
                if !workload.cases[index + 1..].iter().any(|child| {
                    child.step.as_ref().is_some_and(|next| {
                        next.parent.as_deref() == Some(&case.id)
                            && matches!(next.expect, Expected::Json { .. })
                    })
                }) {
                    return Err("tool fixture requires a later factual follow-up parented to its actual call".into());
                }
            }
        }
    }
    Ok(())
}

pub fn settings(workload: &Workload, spec: &WaveSpec) -> RequestSettings {
    let mut settings = workload.request.clone();
    settings.output = workload.request.effective_output(spec.phase).clone();
    settings.warmup_output = None;
    if let Some(step) = workload
        .cases
        .iter()
        .find(|c| Some(c.id.as_str()) == spec.case.as_deref())
        .and_then(|c| c.step.as_ref())
    {
        settings.cache = step.cache;
        settings.stream =
            matches!(workload.version, 5 | 6) || matches!(step.expect, Expected::Json { .. });
    }
    settings
}

struct Messages<'a>(&'a [Rc<Value>]);
impl Serialize for Messages<'_> {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        serializer.collect_seq(self.0.iter().map(Rc::as_ref))
    }
}

#[derive(Serialize)]
struct HistoryRequest<'a> {
    #[serde(flatten)]
    body: Value,
    messages: Messages<'a>,
}

#[derive(Default)]
pub struct State {
    histories: Vec<(String, Vec<Rc<Value>>)>,
    pending: Option<(String, Vec<Rc<Value>>)>,
    next_wave: u32,
    stopped: bool,
    acquisition: Option<crate::acquisition::AcquisitionIdentity>,
    retained_bytes: usize,
    retained_messages: std::collections::HashSet<*const Value>,
}

impl State {
    pub(crate) fn tool_expectation(
        &self,
        workload: &Workload,
        spec: &WaveSpec,
    ) -> Result<Option<ToolExpectation>> {
        if !matches!(workload.version, 5 | 6) || !crate::acquisition::conversation(workload) {
            return Ok(None);
        }
        let case = workload
            .cases
            .iter()
            .find(|case| Some(case.id.as_str()) == spec.case.as_deref())
            .ok_or("unknown sequence case")?;
        let Some(Step {
            expect: Expected::Tool { key, result },
            ..
        }) = &case.step
        else {
            return Ok(None);
        };
        let (pending_case, messages) = self
            .pending
            .as_ref()
            .ok_or("tool expectation requires admitted history")?;
        if pending_case != &case.id || spec.index != self.next_wave {
            return Err("tool expectation differs from admitted step".into());
        }
        let prior_ids = messages
            .iter()
            .filter_map(|message| message.get("tool_calls").and_then(Value::as_array))
            .flatten()
            .filter_map(|call| call.get("id").and_then(Value::as_str))
            .map(str::to_owned)
            .collect();
        Ok(Some(ToolExpectation {
            key: key.clone(),
            prior_ids,
            result: result.clone(),
            history_bytes: serde_json::to_vec(&Messages(messages))
                .map_err(|e| e.to_string())?
                .len(),
            history_messages: messages.len(),
            history_cap: history_cap(workload),
            message_cap: message_cap(workload),
        }))
    }

    pub fn request(
        &mut self,
        context: &crate::wire::BodyContext<'_>,
        spec: &WaveSpec,
        lane: u32,
    ) -> Result<String> {
        if !crate::acquisition::conversation(context.workload) {
            return crate::wire::request_body(context, spec, lane);
        }
        if self.stopped {
            return Err("sequence cannot admit a step after invalid or incomplete evidence".into());
        }
        if spec.index != self.next_wave || self.pending.is_some() || lane != 0 {
            return Err(
                "sequence admission must follow each settled declared step exactly once".into(),
            );
        }
        if context.workload.version == 6 && self.acquisition != spec.acquisition {
            if context.workload.cells.first().map(|c| c.id.as_str()) != Some(spec.cell.as_str()) {
                return Err("acquisition must begin at its first required step".into());
            }
            self.histories.clear();
            self.retained_bytes = 0;
            self.retained_messages.clear();
            self.acquisition = spec.acquisition;
        }
        let case = context
            .workload
            .cases
            .iter()
            .find(|c| Some(c.id.as_str()) == spec.case.as_deref())
            .ok_or("unknown sequence case")?;
        let step = case.step.as_ref().ok_or("missing step")?;
        let mut body: Value =
            serde_json::from_str(&crate::wire::request_body(context, spec, lane)?)
                .map_err(|e| e.to_string())?;
        let inputs = body
            .as_object_mut()
            .and_then(|body| body.remove("messages"))
            .ok_or("missing sequence inputs")?;
        let Value::Array(inputs) = inputs else {
            return Err("sequence inputs must be messages".into());
        };
        let mut messages = if let Some(parent) = &step.parent {
            self.histories
                .iter()
                .find(|(id, _)| id == parent)
                .map(|(_, messages)| messages.clone())
                .ok_or("sequence parent lacks valid retained output")?
        } else {
            Vec::new()
        };
        messages.extend(inputs.into_iter().map(Rc::new));
        if messages.len() > message_cap(context.workload) {
            return Err(if context.workload.version == 6 {
                "sequence accumulated history exceeds declared message bound"
            } else {
                "sequence accumulated history exceeds 64 messages"
            }
            .into());
        }
        let namespace = context
            .cache_namespace
            .ok_or("sequence needs a private cache namespace")?;
        let acquired_namespace = spec
            .acquisition
            .map(|id| crate::acquisition::namespace(namespace, id));
        let namespace = acquired_namespace.as_deref().unwrap_or(namespace);
        body["cache_salt"] = Value::String(format!("{namespace}-{}", step.history));
        if let Expected::Tool { .. } = step.expect {
            body["tools"] = json!([{"type":"function","function":{"name":"lookup_fact","description":"Return the declared local fixture value for one key.","parameters":{"type":"object","properties":{"key":{"type":"string"}},"required":["key"],"additionalProperties":false}}}]);
            body["tool_choice"] = json!({"type":"function","function":{"name":"lookup_fact"}});
        }
        let encoded = serde_json::to_string(&HistoryRequest {
            body,
            messages: Messages(&messages),
        })
        .map_err(|e| e.to_string())?;
        if encoded.len() > input_cap(context.workload) {
            return Err(if context.workload.version == 6 {
                "sequence accumulated input exceeds declared encoded input cap"
            } else {
                "sequence accumulated input exceeds 128 KiB"
            }
            .into());
        }
        self.check_retention(context.workload, &messages)?;
        self.pending = Some((case.id.clone(), messages));
        Ok(encoded)
    }

    pub fn observe(
        &mut self,
        plan: &Plan,
        spec: &WaveSpec,
        attempt: &Attempt,
        body: &[u8],
    ) -> Result<Option<Check>> {
        if !crate::acquisition::conversation(&plan.workload) {
            return Ok(None);
        }
        let case = plan
            .workload
            .cases
            .iter()
            .find(|c| Some(c.id.as_str()) == spec.case.as_deref())
            .ok_or("unknown sequence case")?;
        let step = case.step.as_ref().ok_or("missing step")?;
        let streamed_tool = self
            .tool_expectation(&plan.workload, spec)?
            .map(|expected| crate::wire::sequence_tool(attempt, body, expected))
            .transpose()?
            .flatten();
        let streamed_tool_response = streamed_tool.is_some();
        let (pending_case, mut messages) = self
            .pending
            .take()
            .ok_or("sequence response lacks admitted history")?;
        if pending_case != case.id || spec.index != self.next_wave {
            return Err("sequence response differs from its admitted step".into());
        }
        self.next_wave += 1;
        let mut check = Check {
            history: step.history.clone(),
            parent: step.parent.clone(),
            correct: false,
            canonical_match: None,
            strict_match: None,
            error: None,
        };
        let observed = (|| {
            if attempt.status != Status::Complete {
                return Err("sequence response did not complete".to_string());
            }
            let message = if let Some(message) = streamed_tool {
                message
            } else if settings(&plan.workload, spec).stream {
                let content =
                    crate::wire::sequence_answer(attempt, body, matches!(plan.version, 3..=5))?;
                json!({"role":"assistant","content":content})
            } else {
                let mut response: Value =
                    serde_json::from_slice(body).map_err(|_| "sequence response is not JSON")?;
                response
                    .pointer_mut("/choices/0/message")
                    .ok_or("sequence response lacks message")?
                    .take()
            };
            if message.get("role").and_then(Value::as_str) != Some("assistant") {
                return Err("sequence response role must be assistant".into());
            }
            match &step.expect {
                Expected::Json { value, strict } => {
                    if message.get("tool_calls").is_some_and(|v| !v.is_null()) {
                        return Err("unexpected tool call in factual response".into());
                    }
                    let content = message
                        .get("content")
                        .and_then(Value::as_str)
                        .ok_or("factual response requires text")?;
                    let parsed: Fact = serde_json::from_str(content).map_err(|_| "factual response requires one fact string without duplicate or unknown fields")?;
                    check.strict_match = strict.as_ref().map(|expected| content == expected);
                    check.canonical_match =
                        Some(content == serde_json::to_string(value).map_err(|e| e.to_string())?);
                    check.correct = &parsed == value;
                    if &parsed != value {
                        return Err(
                            "factual response differs from prospective expected object".into()
                        );
                    }
                    if check.strict_match == Some(false) {
                        return Err(
                            "declared strict-output check failed; semantic JSON matched".into()
                        );
                    }
                    messages.push(Rc::new(json!({"role":"assistant","content":content})));
                }
                Expected::Tool { key, result } => {
                    if attempt.finish_reason.as_deref() != Some("tool_calls")
                        && !(streamed_tool_response
                            && attempt.finish_reason.as_deref() == Some("stop"))
                    {
                        return Err(
                            "tool response lacks a supported successful finish reason".into()
                        );
                    }
                    let calls = message
                        .get("tool_calls")
                        .and_then(Value::as_array)
                        .ok_or("expected lookup_fact tool call")?;
                    if calls.len() != 1
                        || message
                            .get("content")
                            .is_some_and(|v| !v.is_null() && v.as_str() != Some(""))
                    {
                        return Err("tool step requires exactly one call and no answer text".into());
                    }
                    let call = &calls[0];
                    let id = call
                        .get("id")
                        .and_then(Value::as_str)
                        .filter(|id| identifier(id))
                        .ok_or("invalid tool call ID")?;
                    if call.get("type").and_then(Value::as_str) != Some("function")
                        || call.pointer("/function/name").and_then(Value::as_str)
                            != Some("lookup_fact")
                    {
                        return Err("unexpected tool type or name".into());
                    }
                    let raw = call
                        .pointer("/function/arguments")
                        .and_then(Value::as_str)
                        .ok_or("tool arguments must be encoded JSON")?;
                    #[derive(Deserialize)]
                    #[serde(deny_unknown_fields)]
                    struct Arguments {
                        key: String,
                    }
                    let args: Arguments = serde_json::from_str(raw)
                        .map_err(|_| "malformed, duplicate or unknown tool arguments")?;
                    if &args.key != key {
                        return Err("tool arguments differ from declared fixture key".into());
                    }
                    if messages.iter().any(|m| {
                        m.get("tool_calls")
                            .and_then(Value::as_array)
                            .is_some_and(|calls| {
                                calls
                                    .iter()
                                    .any(|c| c.get("id").and_then(Value::as_str) == Some(id))
                            })
                    }) {
                        return Err("tool call ID reused within history".into());
                    }
                    messages.push(Rc::new(
                        json!({"role":"assistant","content":null,"tool_calls":calls}),
                    ));
                    messages.push(Rc::new(
                        json!({"role":"tool","tool_call_id":id,"content":result}),
                    ));
                    check.correct = true;
                }
            }
            if messages.len() > message_cap(&plan.workload)
                || serde_json::to_vec(&Messages(&messages))
                    .map_err(|e| e.to_string())?
                    .len()
                    > history_cap(&plan.workload)
            {
                return Err(if plan.workload.version == 6 {
                    "retained sequence history exceeds declared history cap"
                } else {
                    "retained sequence history exceeds 128 KiB"
                }
                .into());
            }
            self.check_retention(&plan.workload, &messages)?;
            Ok(messages)
        })();
        match observed {
            Ok(history) => {
                check.correct = true;
                if eligibility(
                    attempt,
                    &settings(&plan.workload, spec),
                    crate::acquisition::eligibility_phase(&plan.workload, spec.phase),
                )
                .is_empty()
                {
                    if plan.workload.version == 6 {
                        self.retained_bytes = self.retention_size(&history)?;
                        self.retained_messages
                            .extend(history.iter().map(Rc::as_ptr));
                    }
                    self.histories.push((case.id.clone(), history));
                } else {
                    self.stopped = true;
                }
            }
            Err(error) => {
                check.error = Some(error);
                self.stopped = true;
            }
        }
        Ok(Some(check))
    }

    fn check_retention(&self, workload: &Workload, active: &[Rc<Value>]) -> Result<()> {
        if workload.version != 6 {
            return Ok(());
        }
        if self.retention_size(active)? > history_cap(workload) {
            return Err("shared retained history exceeds declared budget".into());
        }
        Ok(())
    }
    fn retention_size(&self, active: &[Rc<Value>]) -> Result<usize> {
        // Each retained Rc payload is charged once across branches, plus every
        // pointer array. Do not reserialize previously retained prefixes.
        let mut bytes = self
            .retained_bytes
            .checked_add(std::mem::size_of_val(active))
            .ok_or("history budget overflow")?;
        for message in active {
            if !self.retained_messages.contains(&Rc::as_ptr(message)) {
                bytes = bytes
                    .checked_add(
                        serde_json::to_vec(message.as_ref())
                            .map_err(|e| e.to_string())?
                            .len(),
                    )
                    .ok_or("history budget overflow")?;
            }
        }
        Ok(bytes)
    }
}
fn input_cap(workload: &Workload) -> usize {
    workload
        .acquisition
        .as_ref()
        .map_or(128 * 1024, crate::acquisition::Protocol::input_bytes)
}
fn history_cap(workload: &Workload) -> usize {
    workload
        .acquisition
        .as_ref()
        .map_or(128 * 1024, crate::acquisition::Protocol::history_bytes)
}
fn message_cap(workload: &Workload) -> usize {
    if workload.version == 6 { 128 * 67 } else { 64 }
}
