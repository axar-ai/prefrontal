use text_classifier::{Classifier, init_logger, BuiltinModel};
use log::{info, error};
use env_logger;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logger();
    info!("=== Starting Text Classifier Demo ===");

    let classifier = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)?
        .add_class("sports", vec![
            "football game",
            "basketball match",
            "tennis tournament"
        ])?
        .add_class("tech", vec![
            "computer programming",
            "software development",
            "artificial intelligence"
        ])?
        .add_class("entertainment", vec![
            "The movie premiere was a huge success",
            "A new album dropped from the famous artist",
            "The TV series finale aired last night"
        ])?
        .add_class("business", vec![
            "The stock market showed significant gains",
            "The company announced a merger deal",
            "Quarterly earnings exceeded expectations"
        ])?
        .add_class("science", vec![
            "Researchers discovered a new species",
            "The experiment yielded interesting results",
            "A breakthrough in quantum computing"
        ])?
        .build()?;

    info!("Classifier built successfully");

    // Test with 10 different inputs
    let test_inputs = vec![
        "The team won the championship game last night",           // sports
        "New smartphone features advanced AI capabilities",        // technology
        "Latest blockbuster movie breaks box office records",     // entertainment
        "Startup raises millions in venture capital funding",     // business
        "Scientists make breakthrough in cancer research",        // science
        "The quarterback threw the winning pass",                 // sports
        "The neural network achieved better accuracy",            // technology
        "The concert was sold out in minutes",                   // entertainment
        "The CEO announced a new business strategy",             // business
        "The space telescope captured amazing images",           // science
    ];

    info!("=== Running Classifications ===\n");
    for (i, text) in test_inputs.iter().enumerate() {
        info!("\nTest {}/{}:", i + 1, test_inputs.len());
        info!("Input: {}", text);
        process_input(&classifier, text)?;
    }

    info!("\n=== Demo Complete ===\n");
    Ok(())
}

fn process_input(classifier: &Classifier, text: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("\nProcessing: {}", text);
    
    let (class, scores) = classifier.predict(text)?;
    
    println!("Predicted class: {}", class);
    println!("Confidence scores:");
    for (label, score) in scores {
        println!("  {}: {:.4}", label, score);
    }

    Ok(())
}
