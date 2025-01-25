use text_classifier::Classifier;
use std::path::PathBuf;

fn get_test_paths() -> (PathBuf, PathBuf) {
    let model_path = PathBuf::from("models/onnx-minilm/model.onnx");
    let tokenizer_path = PathBuf::from("models/onnx-minilm/tokenizer.json");
    (model_path, tokenizer_path)
}

#[test]
fn test_end_to_end_classification() {
    let (model_path, tokenizer_path) = get_test_paths();
    let mut classifier = Classifier::new(
        model_path.to_str().unwrap(),
        tokenizer_path.to_str().unwrap()
    );

    classifier.add_class("sports", vec!["football game", "basketball match"]);
    classifier.build().expect("Build should succeed");

    let (class, scores) = classifier.predict("soccer match");
    
    assert_eq!(class, "sports");
    assert!(scores.contains_key("sports"));
    assert!(scores["sports"] > 0.5);
}

#[test]
fn test_multiple_classes() {
    let (model_path, tokenizer_path) = get_test_paths();
    let mut classifier = Classifier::new(
        model_path.to_str().unwrap(),
        tokenizer_path.to_str().unwrap()
    );

    classifier.add_class("sports", vec!["football game", "basketball match"]);
    classifier.add_class("tech", vec!["computer program", "software code"]);
    classifier.build().expect("Build should succeed");

    let (class_sports, scores_sports) = classifier.predict("soccer match");
    let (class_tech, scores_tech) = classifier.predict("python programming");

    assert_eq!(class_sports, "sports");
    assert_eq!(class_tech, "tech");
    assert!(scores_sports["sports"] > scores_sports["tech"]);
    assert!(scores_tech["tech"] > scores_tech["sports"]);
} 