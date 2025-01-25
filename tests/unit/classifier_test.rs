use text_classifier::{Classifier, BuiltinModel};
use ndarray::Array1;

fn setup_test_classifier() -> Classifier {
    Classifier::builder()
        .with_custom_model(
            "models/onnx-minilm/model.onnx",
            "models/onnx-minilm/tokenizer.json"
        )
        .build()
        .expect("Failed to create classifier")
}

#[test]
fn test_empty_class_handling() {
    let result = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)
        .add_class("empty", vec![])
        .build();
    assert!(result.is_err());
}

#[test]
fn test_unknown_class_prediction() {
    let classifier = setup_test_classifier();
    let (class, scores) = classifier.predict("test").unwrap();
    assert_eq!(class, "unknown");
    assert!(scores.is_empty());
}

#[test]
fn test_tokenizer_padding() {
    let classifier = setup_test_classifier();
    let result = classifier.predict("short text");
    assert!(result.is_ok());
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