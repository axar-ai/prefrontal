use text_classifier::Classifier;

#[test]
fn test_classifier_creation() {
    let mut classifier = Classifier::new(
        "models/onnx-minilm/model.onnx",
        "models/onnx-minilm/tokenizer.json"
    );
    
    classifier.add_class("test", vec!["test example"]);
    let result = classifier.build();
    assert!(result.is_ok(), "Classifier should build successfully");
}

#[test]
fn test_add_class() {
    let mut classifier = Classifier::new(
        "models/onnx-minilm/model.onnx",
        "models/onnx-minilm/tokenizer.json"
    );
    
    classifier.add_class("test", vec!["example one", "example two"]);
    classifier.build().expect("Failed to build classifier");
    
    let (class, scores) = classifier.predict("test example")
        .expect("Failed to make prediction");
        
    assert_eq!(class, "test");
    assert!(scores.contains_key("test"));
}

#[test]
fn test_end_to_end_classification() {
    let mut classifier = Classifier::new(
        "models/onnx-minilm/model.onnx",
        "models/onnx-minilm/tokenizer.json"
    );
    
    classifier.add_class("sports", vec!["football game", "basketball match"]);
    classifier.build().expect("Failed to build classifier");
    
    let (class, scores) = classifier.predict("soccer match")
        .expect("Failed to make prediction");
        
    assert_eq!(class, "sports");
    assert!(scores.contains_key("sports"));
}

#[test]
fn test_multiple_classes() {
    let mut classifier = Classifier::new(
        "models/onnx-minilm/model.onnx",
        "models/onnx-minilm/tokenizer.json"
    );
    
    classifier.add_class("sports", vec!["football game", "basketball match"]);
    classifier.add_class("technology", vec!["python coding", "software development"]);
    classifier.build().expect("Failed to build classifier");
    
    let (class_sports, scores_sports) = classifier.predict("soccer match")
        .expect("Failed to make prediction");
    let (class_tech, scores_tech) = classifier.predict("python programming")
        .expect("Failed to make prediction");
        
    assert_eq!(class_sports, "sports");
    assert_eq!(class_tech, "technology");
    assert!(scores_sports.contains_key("sports"));
    assert!(scores_tech.contains_key("technology"));
} 