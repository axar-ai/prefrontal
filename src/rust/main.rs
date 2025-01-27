use prefrontal::{Classifier, BuiltinModel, ClassDefinition, ModelManager};
use log::info;
use env_logger;
use clap::Parser;
use std::time::Instant;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Force a fresh download of the model files
    #[arg(short, long)]
    fresh: bool,
}

async fn ensure_model_downloaded(fresh: bool) -> Result<(), Box<dyn std::error::Error>> {
    let manager = ModelManager::new_default()?;
    let model = BuiltinModel::MiniLM;

    if fresh {
        info!("Fresh download requested - removing any existing model files...");
        manager.remove_download(model)?;
    }
    
    if !manager.is_model_downloaded(model) {
        info!("Downloading model...");
        manager.download_model(model).await?;
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args = Args::parse();
    
    info!("=== Starting Text Classifier Demo ===");

    // Ensure model is downloaded before proceeding
    ensure_model_downloaded(args.fresh).await?;

    let start_time = Instant::now();
    info!("Building classifier...");

    // Create classifier with comprehensive categories
    let classifier = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)?
        .add_class(
            ClassDefinition::new(
                "technology",
                "Technology, software, hardware, and digital innovation"
            ).with_examples(vec![
                // Software & Development
                "New programming language features improve developer productivity",
                "Latest software update includes critical security patches",
                "Open source project reaches one million users",
                "Cloud platform announces serverless computing features",
                
                // Hardware & Devices
                "Smartphone manufacturer reveals foldable device",
                "New processor achieves breakthrough in quantum computing",
                "Revolutionary battery technology extends device life",
                
                // AI & Machine Learning
                "AI model achieves human-level performance in medical diagnosis",
                "Machine learning system improves autonomous driving",
                "Neural network breakthrough in language translation",
                
                // Cybersecurity
                "Researchers discover critical security vulnerability",
                "New encryption method provides quantum resistance",
                "Cybersecurity firm prevents major ransomware attack"
            ])
        )?
        .add_class(
            ClassDefinition::new(
                "science",
                "Scientific research, discoveries, and breakthroughs"
            ).with_examples(vec![
                // Space & Astronomy
                "Astronomers discover Earth-like exoplanet in habitable zone",
                "Space telescope captures black hole merger",
                "Mars rover finds evidence of ancient water",
                
                // Medicine & Biology
                "Researchers develop new cancer treatment method",
                "Gene therapy shows promise in treating rare disease",
                "Brain study reveals new insights into memory formation",
                
                // Physics & Chemistry
                "Particle accelerator discovers new elementary particle",
                "Quantum teleportation achieved at record distance",
                "New material shows promise for clean energy storage",
                
                // Environmental Science
                "Climate study predicts sea level changes",
                "Research shows impact of microplastics on marine life",
                "New method developed for carbon capture"
            ])
        )?
        .add_class(
            ClassDefinition::new(
                "business",
                "Business, finance, economics, and corporate news"
            ).with_examples(vec![
                // Corporate News
                "Company announces major acquisition deal",
                "Startup raises record-breaking funding round",
                "CEO steps down amid strategic restructuring",
                
                // Markets & Trading
                "Stock market reaches historic milestone",
                "Cryptocurrency prices surge on regulatory news",
                "Oil prices affect global energy markets",
                
                // Economy & Policy
                "Central bank adjusts interest rates",
                "Trade agreement impacts global commerce",
                "Economic indicators show growth trends",
                
                // Industry & Innovation
                "Electric vehicle maker expands production",
                "Renewable energy sector creates jobs",
                "Supply chain innovations reduce costs"
            ])
        )?
        .add_class(
            ClassDefinition::new(
                "health",
                "Health, wellness, medicine, and healthcare"
            ).with_examples(vec![
                // Medical Research
                "Clinical trial shows promising results for vaccine",
                "New treatment method reduces recovery time",
                "Study reveals link between diet and longevity",
                
                // Mental Health
                "Research highlights importance of work-life balance",
                "New therapy approach helps anxiety patients",
                "Study shows impact of social media on mental health",
                
                // Public Health
                "Health officials release new dietary guidelines",
                "Pandemic response measures show effectiveness",
                "Environmental factors impact public health",
                
                // Healthcare Technology
                "AI system improves medical diagnosis accuracy",
                "Telemedicine platform expands healthcare access",
                "Wearable devices monitor vital signs"
            ])
        )?
        .add_class(
            ClassDefinition::new(
                "education",
                "Education, learning, teaching, and academic developments"
            ).with_examples(vec![
                // Teaching Methods
                "New teaching approach improves student engagement",
                "Online learning platform launches innovative features",
                "Study shows benefits of project-based learning",
                
                // Educational Technology
                "AI tutor provides personalized learning experience",
                "Virtual reality enhances classroom instruction",
                "Digital tools improve student assessment",
                
                // Academic Research
                "Study reveals factors in student success",
                "Research shows impact of early childhood education",
                "New curriculum improves learning outcomes",
                
                // Educational Policy
                "School system implements new learning standards",
                "Higher education faces enrollment changes",
                "Education reform addresses equity issues"
            ])
        )?
        .add_class(
            ClassDefinition::new(
                "environment",
                "Environmental issues, sustainability, and climate"
            ).with_examples(vec![
                // Climate Change
                "Global temperature records show warming trend",
                "Arctic ice levels reach historic low",
                "Climate models predict future scenarios",
                
                // Conservation
                "Conservation efforts save endangered species",
                "Marine sanctuary protects coral reefs",
                "Forest restoration project shows success",
                
                // Sustainability
                "Renewable energy adoption accelerates",
                "Zero-waste initiatives reduce plastic use",
                "Sustainable agriculture methods increase yields",
                
                // Environmental Policy
                "Nations agree on emissions reduction targets",
                "New regulations protect endangered habitats",
                "Cities implement green infrastructure"
            ])
        )?
        .add_class(
            ClassDefinition::new(
                "culture",
                "Arts, entertainment, media, and cultural trends"
            ).with_examples(vec![
                // Arts & Entertainment
                "Film festival celebrates independent cinema",
                "Artist's exhibition breaks attendance records",
                "Music streaming transforms industry",
                
                // Media & Publishing
                "Digital media platform launches new format",
                "Journalism adapts to changing landscape",
                "Podcast series explores cultural issues",
                
                // Cultural Trends
                "Social media influences cultural change",
                "Virtual events reshape entertainment",
                "Gaming culture impacts mainstream media",
                
                // Creative Industries
                "Virtual reality revolutionizes art creation",
                "Digital platforms support independent artists",
                "Creative industry adapts to remote work"
            ])
        )?
        .build()?;

    let build_time = start_time.elapsed();
    info!("=== Classifier Built Successfully (took {:.2?}) ===\n", build_time);
    
    let test_inputs = vec![
        // Technology examples
        "Developers embrace new programming paradigm for cloud computing",
        "Quantum computer achieves breakthrough in optimization problems",
        
        // Science examples
        "Astronomers detect mysterious signals from distant galaxy",
        "Breakthrough in CRISPR gene editing shows promise for treating genetic disorders",
        
        // Business examples
        "Tech startup revolutionizes supply chain management with blockchain",
        "Global markets react to unexpected economic indicators",
        
        // Health examples
        "New research reveals connection between gut bacteria and immune system",
        "Mental health platform provides AI-powered therapy sessions",
        
        // Education examples
        "Universities adopt hybrid learning models for greater accessibility",
        "Study shows impact of gamification on student engagement",
        
        // Environment examples
        "Innovative carbon capture technology shows promising results",
        "Coastal cities implement nature-based solutions for climate resilience",
        
        // Culture examples
        "Virtual reality art exhibition breaks boundaries of traditional galleries",
        "Streaming platform's original content reshapes entertainment industry",
        
        // Cross-domain examples
        "AI-powered environmental monitoring system helps conservation efforts",
        "Digital health platform combines telemedicine with AI diagnostics",
        "Educational technology startup receives major investment",
        "Cultural shift towards sustainable lifestyle impacts consumer behavior"
    ];

    info!("=== Running Classifications ({} inputs) ===\n", test_inputs.len());
    let classify_start = std::time::Instant::now();
    
    for (i, text) in test_inputs.iter().enumerate() {
        info!("\nTest {}/{} (elapsed: {:.2?}):", i + 1, test_inputs.len(), classify_start.elapsed());
        process_input(&classifier, text)?;
    }

    let total_time = start_time.elapsed();
    let classify_time = classify_start.elapsed();
    
    info!("\n=== Demo Complete ===");
    info!("Total time: {:.2?}", total_time);
    info!("Build time: {:.2?}", build_time);
    info!("Classification time: {:.2?}", classify_time);
    info!("Average time per classification: {:.2?}", classify_time / test_inputs.len() as u32);
    
    Ok(())
}

fn process_input(classifier: &Classifier, text: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Truncate text for display (show first 60 chars + "..." if longer)
    let display_text = if text.len() > 60 {
        format!("{}...", &text[..60])
    } else {
        text.to_string()
    };
    info!("\nProcessing: {}", display_text);
    
    match classifier.predict(text) {
        Ok((class, scores)) => {
            let mut scores: Vec<_> = scores.into_iter().collect();
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            
            println!("\nResults for: {}", display_text);
            println!("  Predicted class: {}", class);
            println!("  Confidence scores (sorted):");
            for (label, score) in scores {
                println!("    {}: {:.1}%", label, score * 100.0);
            }
        }
        Err(e) => {
            eprintln!("\nError processing text: {}", e);
            eprintln!("Input text: {}", display_text);
            eprintln!("Consider:");
            eprintln!("  - Checking if the text is empty");
            eprintln!("  - Splitting long text into smaller chunks (max 256 tokens)");
            eprintln!("  - Ensuring the text is valid UTF-8");
            return Err(e.into());
        }
    }

    Ok(())
}
