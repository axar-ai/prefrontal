use text_classifier::Classifier;
use ndarray::Array1;

fn setup_test_classifier() -> Classifier {
    let model_path = "models/onnx-minilm/model.onnx";
    let tokenizer_path = "models/onnx-minilm/tokenizer.json";
    Classifier::new(model_path, tokenizer_path)
}

#[test]
fn test_empty_class_handling() {
    let mut classifier = setup_test_classifier();
    classifier.add_class("empty", vec![]);
    let result = classifier.build();
    assert!(result.is_ok());
}

#[test]
fn test_unknown_class_prediction() {
    let mut classifier = setup_test_classifier();
    let (class, scores) = classifier.predict("test");
    assert_eq!(class, "unknown");
    assert!(scores.is_empty());
}

#[test]
fn test_tokenizer_padding() {
    let classifier = setup_test_classifier();
    if let Some(tokens) = classifier.tokenize("short text") {
        assert!(tokens.len() >= 2);  // At least 2 tokens
        assert!(tokens.iter().any(|&x| x == 0));  // Should contain padding
    }
}

#[test]
fn test_build_idempotent() {
    let mut classifier = setup_test_classifier();
    classifier.add_class("test", vec!["example"]);
    
    let first_build = classifier.build();
    let second_build = classifier.build();
    
    assert!(first_build.is_ok());
    assert!(second_build.is_ok());
}

#[test]
fn test_normalize_vector() {
    let vec = Array1::from_vec(vec![3.0, 4.0]);  // 3-4-5 triangle
    let normalized = Classifier::normalize_vector(&vec);
    
    assert!((normalized[0] - 0.6).abs() < 1e-6);  // 3/5
    assert!((normalized[1] - 0.8).abs() < 1e-6);  // 4/5
}

// ... rest of the unit tests from src/classifier.rs ... 