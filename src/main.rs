use text_classifier::{Classifier, BuiltinModel};
use log::info;
use env_logger;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    info!("=== Starting Text Classifier Demo ===");

    let classifier = Classifier::builder()
        .with_model(BuiltinModel::MiniLM)?
        .add_class("sports", vec![
            // Team Sports
            "The team scored in the final minutes of the championship game",
            "Local basketball team advances to state finals",
            "Soccer match ends in dramatic penalty shootout",
            "Baseball team clinches division title with walk-off home run",
            "Hockey team wins overtime thriller on power play goal",
            "Rugby team dominates in international tournament",
            "Volleyball team secures spot in national championships",
            
            // Individual Sports
            "Tennis player wins her third grand slam title",
            "Gymnast performs perfect routine at Olympics",
            "Swimmer breaks world record in freestyle event",
            "Boxer wins heavyweight championship bout",
            "Golfer sinks dramatic putt to win major tournament",
            "Marathon runner sets new course record",
            "Figure skater lands historic quadruple jump",
            
            // Sports Events/News
            "Athletes prepare for upcoming Olympic trials",
            "Record-breaking performance at track meet",
            "Sports league announces expansion teams",
            "Coach wins prestigious award for season success",
            "Stadium renovation completed ahead of season",
            "Draft pick signs record-breaking rookie contract",
            "Injury report updates team's playoff chances"
        ])?
        .add_class("tech", vec![
            // AI and Machine Learning
            "New artificial intelligence model breaks performance records",
            "Machine learning algorithm improves medical diagnosis",
            "Neural network achieves human-level performance",
            "AI system masters complex strategy game",
            "Deep learning model enhances image recognition",
            
            // Hardware and Infrastructure
            "Startup launches revolutionary quantum computing platform",
            "New processor chip doubles computing speed",
            "Revolutionary memory technology increases storage capacity",
            "Data center expands renewable energy usage",
            "5G network rollout reaches major milestone",
            
            // Software and Security
            "Latest smartphone features advanced security capabilities",
            "Software update improves system performance and stability",
            "Cybersecurity team prevents sophisticated attack",
            "Cloud platform introduces new developer tools",
            "Operating system patch fixes critical vulnerabilities",
            
            // Emerging Technologies
            "Autonomous vehicles complete extensive road tests",
            "Virtual reality system enhances remote collaboration",
            "Blockchain technology revolutionizes supply chain",
            "Robotics company unveils household assistant",
            "Biotech startup develops new gene editing tool",
            
            // Industry News
            "Tech company announces groundbreaking innovation",
            "Research lab achieves quantum supremacy",
            "Startup receives patent for innovative technology",
            "Major tech acquisition reshapes industry landscape",
            "Technology standard approved by international committee"
        ])?
        .add_class("entertainment", vec![
            // Movies
            "Blockbuster movie breaks opening weekend records",
            "Film director wins prestigious award for latest work",
            "Superhero movie sequel announces star-studded cast",
            "Independent film receives festival recognition",
            "Animation studio reveals upcoming movie lineup",
            
            // Music
            "Popular band announces world tour dates",
            "Artist's new album tops global charts",
            "Music festival draws massive crowd attendance",
            "Singer wins multiple awards at ceremony",
            "Historic concert venue reopens with special performance",
            
            // Television
            "New streaming series receives critical acclaim",
            "Reality show finale draws record viewership",
            "Television network announces fall lineup",
            "Documentary series wins Emmy award",
            "Streaming platform launches original content",
            
            // Gaming
            "Video game release exceeds sales expectations",
            "Esports tournament attracts millions of viewers",
            "Gaming company reveals next-generation console",
            "Mobile game becomes global phenomenon",
            "Virtual gaming event unites players worldwide",
            
            // Arts and Culture
            "Theater production receives standing ovation",
            "Art exhibition showcases emerging talents",
            "Celebrity hosts star-studded charity gala",
            "Broadway show announces international tour",
            "Cultural festival celebrates diverse artists"
        ])?
        .add_class("business", vec![
            // Corporate Finance
            "Company reports record quarterly earnings",
            "Stock market reaches all-time high",
            "Investment firm announces major acquisition",
            "Corporation issues successful public offering",
            "Company stock splits after sustained growth",
            
            // Startups and Funding
            "Startup secures major venture capital funding",
            "Angel investors back promising new company",
            "Tech startup reaches unicorn valuation",
            "Crowdfunding campaign exceeds target goal",
            "Incubator program launches new businesses",
            
            // Market and Economy
            "Economic indicators show strong growth",
            "Market analysts predict positive outlook",
            "Trade agreement boosts international commerce",
            "Currency exchange rates impact global trade",
            "Consumer confidence index shows improvement",
            
            // Corporate Strategy
            "Merger creates new industry powerhouse",
            "Company launches innovative business model",
            "Retail chain expands international presence",
            "Business restructuring improves efficiency",
            "Strategic partnership enhances market position",
            
            // Industry Trends
            "Industry disruption leads to market shifts",
            "Small business growth exceeds expectations",
            "Digital transformation drives business success",
            "Supply chain innovations reduce costs",
            "Sustainable business practices gain momentum"
        ])?
        .add_class("science", vec![
            // Space and Astronomy
            "Space telescope captures unprecedented cosmic images",
            "Astronomers discover new exoplanet system",
            "Mars rover finds evidence of ancient water",
            "Scientists observe gravitational wave event",
            "Space station conducts groundbreaking experiment",
            
            // Biology and Medicine
            "Researchers discover new species in remote region",
            "Clinical trials show promising treatment results",
            "Genetic research reveals evolutionary insights",
            "Marine biologists track rare ocean phenomenon",
            "Medical breakthrough offers new treatment hope",
            
            // Physics and Technology
            "Particle accelerator achieves new milestone",
            "Quantum experiment challenges physics theories",
            "Scientists develop revolutionary material",
            "Fusion reactor reaches temperature record",
            "Researchers advance quantum computing theory",
            
            // Environmental Science
            "Study reveals climate change impacts",
            "Research shows ocean acidification effects",
            "Scientists track atmospheric changes",
            "Environmental study maps biodiversity",
            "Research documents glacier movements",
            
            // Other Scientific Fields
            "Archaeological dig uncovers ancient civilization",
            "Chemical analysis reveals new compound",
            "Geological survey finds mineral deposits",
            "Anthropologists document cultural practices",
            "Paleontologists discover dinosaur fossils"
        ])?
        .build()?;

    info!("Classifier built successfully");

    // Test with diverse inputs
    let test_inputs = vec![
        // Sports tests
        "The team won the championship game last night",
        "The quarterback threw the winning pass",
        "Athletes set new records at the competition",
        
        // Tech tests
        "New smartphone features advanced AI capabilities",
        "The neural network achieved better accuracy",
        "Developers release major software update",
        
        // Entertainment tests
        "Latest blockbuster movie breaks box office records",
        "The concert was sold out in minutes",
        "New video game launches to critical acclaim",
        
        // Business tests
        "Startup raises millions in venture capital funding",
        "The CEO announced a new business strategy",
        "Company expands into international markets",
        
        // Science tests
        "Scientists make breakthrough in cancer research",
        "The space telescope captured amazing images",
        "Researchers discover new marine species"
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
