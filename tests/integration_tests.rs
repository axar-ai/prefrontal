use text_classifier::{Classifier, BuiltinModel};
use env_logger::{Builder, Env};

// Initialize test logger
fn init() {
    let _ = Builder::from_env(Env::default().default_filter_or("warn"))
        .try_init();
}

#[test]
fn test_basic_classification() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)
        .add_class("test", vec!["test example"])
        .build()?;
    
    let (class, scores) = classifier.predict("test example")?;
    assert_eq!(class, "test");
    assert!(scores["test"] > 0.5);
    Ok(())
}

#[test]
fn test_multiple_examples() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)
        .add_class("test", vec!["example one", "example two"])
        .build()?;

    let (class, scores) = classifier.predict("test example")?;
    assert_eq!(class, "test");
    assert!(scores["test"] > 0.5);
    Ok(())
}

#[test]
fn test_multiple_classes() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)
        .add_class("sports", vec!["football game", "basketball match"])
        .add_class("tech", vec!["computer program", "software code"])
        .build()?;

    let (class_sports, scores_sports) = classifier.predict("soccer match")?;
    let (class_tech, scores_tech) = classifier.predict("python programming")?;

    assert_eq!(class_sports, "sports");
    assert_eq!(class_tech, "tech");
    assert!(scores_sports["sports"] > scores_sports["tech"]);
    assert!(scores_tech["tech"] > scores_tech["sports"]);
    Ok(())
}

#[test]
fn test_validation_errors() {
    // Test custom model path validation
    assert!(Classifier::builder()
        .with_custom_model("", "tokenizer.json")
        .build()
        .is_err());
    assert!(Classifier::builder()
        .with_custom_model("model.onnx", "")
        .build()
        .is_err());
    
    // Test empty class validation
    let result = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)
        .add_class("", vec!["example"])
        .build();
    assert!(result.is_err());

    // Test empty examples validation
    let result = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)
        .add_class("label", vec![])
        .build();
    assert!(result.is_err());

    // Test empty example text validation
    let result = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)
        .add_class("label", vec![""])
        .build();
    assert!(result.is_err());
}

#[test]
fn test_builtin_model_characteristics() {
    let model = BuiltinModel::MiniLM;
    let chars = model.characteristics();
    assert_eq!(chars.embedding_size, 384);
    assert_eq!(chars.max_sequence_length, 256);
    assert_eq!(chars.model_size_mb, 85);
}

#[test]
fn test_classifier_info() -> Result<(), Box<dyn std::error::Error>> {
    init();
    let classifier = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)
        .add_class("sports", vec!["football game", "basketball match"])
        .add_class("tech", vec!["computer program", "software code"])
        .build()?;

    let info = classifier.info();
    
    // Test all fields from the info struct
    assert!(!info.model_path.is_empty());
    assert!(!info.tokenizer_path.is_empty());
    assert_eq!(info.num_classes, 2);
    assert!(info.class_labels.contains(&"sports".to_string()));
    assert!(info.class_labels.contains(&"tech".to_string()));
    assert_eq!(info.embedding_size, 384); // MiniLM's embedding size

    Ok(())
} 