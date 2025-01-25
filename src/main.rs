use text_classifier::{Classifier, init_logger};
use log::{info, error};

fn main() {
    init_logger();
    info!("=== Starting Text Classifier Demo ===");

    // Create classifier
    let mut classifier = Classifier::new(
        "models/onnx-minilm/model.onnx",
        "models/onnx-minilm/tokenizer.json"
    );

    // Add 5 different classes with examples
    info!("Adding classes with examples:");
    
    // Sports class
    classifier.add_class("sports", vec![
        "He scored a touchdown in the football game",
        "The basketball match ended in a buzzer-beater",
        "They won the soccer championship"
    ]);

    // Technology class
    classifier.add_class("technology", vec![
        "The new software update improves performance",
        "Developers released a patch for the bug",
        "AI models are getting more sophisticated"
    ]);

    // Entertainment class
    classifier.add_class("entertainment", vec![
        "The movie premiere was a huge success",
        "A new album dropped from the famous artist",
        "The TV series finale aired last night"
    ]);

    // Business class
    classifier.add_class("business", vec![
        "The stock market showed significant gains",
        "The company announced a merger deal",
        "Quarterly earnings exceeded expectations"
    ]);

    // Science class
    classifier.add_class("science", vec![
        "Researchers discovered a new species",
        "The experiment yielded interesting results",
        "A breakthrough in quantum computing"
    ]);

    // Build classifier
    info!("Building classifier...");
    match classifier.build() {
        Ok(_) => info!("Build successful"),
        Err(e) => {
            error!("Build failed: {}", e);
            return;
        }
    }

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
        process_input(&classifier, text);
    }

    info!("\n=== Demo Complete ===\n");
}

fn process_input(classifier: &Classifier, text: &str) {
    println!("\nProcessing: {}", text);
    
    match classifier.predict(text) {
        Ok((class, scores)) => {
            println!("Predicted class: {}", class);
            println!("Confidence scores:");
            for (label, score) in scores {
                println!("  {}: {:.4}", label, score);
            }
        },
        Err(e) => {
            eprintln!("Error making prediction: {}", e);
        }
    }
}
