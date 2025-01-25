use text_classifier::Classifier;

fn setup_classifier() -> Classifier {
    Classifier::new(
        "models/onnx-minilm/model.onnx",
        "models/onnx-minilm/tokenizer.json"
    )
}

#[test]
fn test_demo_with_different_input() {
    let mut classifier = setup_classifier();
    classifier.add_class("sports", vec!["football game", "basketball match"]);
    classifier.build().expect("Build should succeed");
    
    let (class, _) = classifier.predict("chess tournament");
    assert_eq!(class, "sports");  // Should still classify as sports
} 