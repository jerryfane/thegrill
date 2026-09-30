use super::*;

#[test]
fn captures_pair_lanes_by_acquisition() {
    let temp = Temp::new();
    // Each capture sends 8 acquisitions of 1 warmup and 3 measured requests in order;
    // request 54 is the check capture's acquisition 5, measured trial 1.
    let server = Server::new(|stream, index, _| {
        let content = if index == 54 { "1 3" } else { "1 2" };
        respond(stream, content, Some(400), false, 4);
    });
    let before = deployment(&temp, "before.json", "before");
    let after = deployment(&temp, "after.json", "after");
    assert!(baseline(&temp, &server, &before, "a").status.success());
    assert_ne!(
        decoded(&check(&temp, "a", &after, "b"))["result"],
        "INVALID"
    );
    let output = command()
        .arg("outputs")
        .arg(temp.path("a"))
        .arg(temp.path("b"))
        .arg("--json")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    let report = decoded(&output);
    assert_eq!(
        [
            &report["identical"],
            &report["differs"],
            &report["unavailable"]
        ],
        [23, 1, 0]
    );
    let lane = report["lanes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|lane| lane["status"] == "differs")
        .unwrap();
    assert_eq!([&lane["acquisition"], &lane["trial"]], [5, 1]);
    assert_eq!([&lane["char_offset"], &lane["byte_offset"]], [2, 2]);

    let mixed = command()
        .arg("outputs")
        .arg(temp.path("a"))
        .arg(temp.path("a/acquisition-00"))
        .output()
        .unwrap();
    assert_eq!(mixed.status.code(), Some(1));
}
