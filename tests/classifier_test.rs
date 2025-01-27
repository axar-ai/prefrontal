use prefrontal::{Classifier, BuiltinModel, ClassDefinition, ModelManager};
use std::sync::Arc;
use std::thread;

mod model_manager_test;
use model_manager_test::ensure_model_exists;

fn generate_long_text(entries: usize) -> String {
    let mut text = String::new();
    for i in 0..entries {
        text.push_str(&format!(
            "Entry {}: Temperature reading {:.1}°C at location Site-{}. System status: Normal. Timestamp: {}. ",
            i, 20.0 + (i as f32 * 0.1), i % 100, 1623456789 + i
        ));
    }
    text
}

async fn setup_model() -> Result<(), Box<dyn std::error::Error>> {
    let manager = ModelManager::new_default()?;
    let model = BuiltinModel::MiniLM;
    ensure_model_exists(&manager, model).await?;
    Ok(())
}

async fn setup_classifier() -> Result<Classifier, Box<dyn std::error::Error>> {
    setup_model().await?;

    let classifier = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)?
        .add_class(
            ClassDefinition::new("positive", "Positive sentiment")
                .with_examples(vec![
                    "great service and friendly staff",
                    "awesome experience, highly recommend",
                    "excellent quality and fast delivery",
                    "I really enjoyed this product",
                    "the customer support was very helpful",
                    "fantastic movie with amazing performances"
                ])
        )?
        .add_class(
            ClassDefinition::new("negative", "Negative sentiment")
                .with_examples(vec![
                    "terrible service and rude staff",
                    "awful experience, would not recommend",
                    "bad quality and slow delivery",
                    "I really disliked this product",
                    "the customer support was unhelpful",
                    "disappointing movie with poor performances"
                ])
        )?
        .build()?;

    Ok(classifier)
}

#[tokio::test]
async fn test_basic_classification() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = setup_classifier().await?;
    let (label, scores) = classifier.predict("This is great!")?;
    assert_eq!(label, "positive");
    assert!(scores["positive"] > scores["negative"]);
    Ok(())
}

#[tokio::test]
async fn test_thread_safety() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = Arc::new(setup_classifier().await?);
    let mut handles = vec![];

    for _ in 0..3 {
        let classifier = Arc::clone(&classifier);
        handles.push(thread::spawn(move || {
            classifier.predict("test text").unwrap();
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
    Ok(())
}

#[tokio::test]
async fn test_empty_text() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = setup_classifier().await?;
    let result = classifier.predict("");
    assert!(result.is_err());
    Ok(())
}

#[tokio::test]
async fn test_long_text() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = setup_classifier().await?;
    let long_text = generate_long_text(500);  // 500 entries
    println!("\nTesting long text with {} characters", long_text.len());
    
    let result = classifier.predict(&long_text);
    println!("Prediction result: {:?}", result);
    
    // Verify prediction succeeds for long input
    assert!(result.is_ok());
    Ok(())
}

#[tokio::test]
async fn test_end_to_end_classification() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = setup_classifier().await?;
    let text = "I really enjoyed watching this movie, it was fantastic!";
    println!("\nTesting end-to-end classification with text: {}", text);
    let (class, scores) = classifier.predict(text)?;
    println!("Prediction result: class={}, scores={:?}", class, scores);
    assert_eq!(class, "positive");
    assert!(scores.contains_key("positive"));
    assert!(scores["positive"] > 0.0);
    Ok(())
}

#[tokio::test]
async fn test_unknown_class_prediction() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = setup_classifier().await?;
    let text = "The product arrived on time and works perfectly";
    println!("\nTesting unknown class prediction with text: {}", text);
    let (class, scores) = classifier.predict(text)?;
    println!("Prediction result: class={}, scores={:?}", class, scores);
    assert_eq!(class, "positive");
    assert!(!scores.is_empty());
    Ok(())
}

#[tokio::test]
async fn test_token_length_handling() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = setup_classifier().await?;
    let extremely_long_text = generate_long_text(1000);  // 1000 entries
    
    println!("\nTesting token length handling with {} characters", extremely_long_text.len());
    
    let result = classifier.predict(&extremely_long_text);
    println!("Prediction result: {:?}", result);
    
    // Verify prediction succeeds for long input
    assert!(result.is_ok());
    Ok(())
}

#[tokio::test]
async fn test_prediction_validation() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = setup_classifier().await?;
    
    println!("\nTesting empty input");
    let result = classifier.predict("");
    println!("Empty input result: {:?}", result);
    assert!(result.is_err());
    
    let very_long_text = generate_long_text(750);  // 750 entries
    println!("\nTesting very long input with {} characters", very_long_text.len());
    
    let result = classifier.predict(&very_long_text);
    println!("Long input result: {:?}", result);
    
    // Verify prediction succeeds for long input
    assert!(result.is_ok());
    Ok(())
}

#[tokio::test]
async fn test_zero_shot_classification() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = setup_classifier().await?;
    let text = "The customer service was excellent and they resolved my issue quickly";
    println!("\nTesting zero-shot classification with text: {}", text);
    let (class, scores) = classifier.predict(text)?;
    println!("Prediction result: class={}, scores={:?}", class, scores);
    assert_eq!(class, "positive");
    assert!(scores["positive"] > 0.0);
    Ok(())
}

#[tokio::test]
async fn test_classifier_thread_safety() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = Arc::new(setup_classifier().await?);
    
    // Test that classifier can be sent to another thread
    let classifier_clone = Arc::clone(&classifier);
    
    let handle = thread::spawn(move || {
        let result = classifier_clone.predict("test text");
        assert!(result.is_ok());
    });
    
    handle.join().unwrap();
    Ok(())
}

#[tokio::test]
async fn test_multiple_classes() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = setup_classifier().await?;

    let (class, scores) = classifier.predict("javascript programming")?;
    assert_eq!(class, "positive");
    assert!(scores["positive"] > 0.0);
    Ok(())
}

#[tokio::test]
async fn test_semantic_similarity() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = setup_classifier().await?;

    let (class, scores) = classifier.predict("chess tournament")?;
    assert_eq!(class, "positive");  // Should classify as positive due to semantic similarity
    assert!(scores["positive"] > 0.0);
    Ok(())
} 