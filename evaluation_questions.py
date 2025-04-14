"""Evaluation questions for social media ad assessment."""

EVALUATION_QUESTIONS = {
    "creativity": 
    # [
    #     "Does the ad script immediately grab attention within the first few words/visual description?", #checking hook
    #     "Is the central concept or idea behind the ad novel and not commonly seen in similar advertisements?", #checking originality
    #     "Does the execution of the idea feel fresh ,imaginative and innovative?",
    #     "Does the ad evoke a specific emotion (joy, surprise, curiosity, etc.)?",
    #     "Does the script present the product/service in a unique or unconventional way?",
    #     "Does the ad leave a lasting impression or make the viewer think?",
    #     "Does the script avoid clichés and predictable storytelling?",
    #     "Does the ad offer a creative solution to a problem or fulfill a desire in an original manner?",
    #     "Does the overall approach demonstrate a high level of imaginative thinking?",
    #     "Does the ad's opening create an immediate visual or emotional impact?",
    #     "Does the ad effectively use the first few seconds to interrupt the audience’s scrolling behavior and include an unexpected or intriguing element that creates a strong hook, compelling the audience to want to learn more?",
    #     "Does the ad use any unexpected elements (humor, surprise, etc.)?",
    #     "Does the script avoid clichés and overused phrases?",
    #     "Does the ad leave the viewer wanting to know more?",
    #     "Does the ad tell a mini-story or present a scenario that's engaging?",
    #     "Is the ad's call to action creative and compelling, not just standard?",
    #     "Does the overall ad feel memorable and distinct?",
    #     "Does the script effectively use visuals or imagery to enhance creative impact?",
    #     "Does the ad stand out from typical Instagram ad formats?",
    #     "Does the ad use unexpected elements or juxtapositions to spark curiosity or intrigue in the viewer?"
    # ],
    [
        "Does the ad grab attention in the first few seconds?",
        # "Is the central idea of the ad unique compared to other ads in the same category?",
        "Is the execution of the ad creative and innovative?",
        "Does the ad evoke a specific emotion (e.g., joy, surprise, curiosity)?",
        "Does the script avoid clichés, predictable storytelling, and overused phrases?",
        "Does the ad offer a creative solution to a problem or present something in an original way?",
        "Is the ad memorable and distinct, leaving the viewer wanting to know more?",
        "Is the call to action in the ad creative and compelling?",
        "Does the ad effectively use visuals or imagery to enhance its impact?",
        # "Does the ad stand out from typical Instagram ads?",
        "Does the ad use unexpected elements (e.g., humor, surprise) to create curiosity or intrigue?"
    ],
    # "emotional_appeal": [
    #     "Does the ad tap into a core emotional need or aspiration?",
    #     "Does the content create a clear vision of potential transformation or benefit?",
    #     "Are the emotional benefits of the product/service prominently highlighted?",
    #     "Does the ad make the audience feel understood or validated?",
    #     "Does the ad evoke a positive emotion (e.g., joy, excitement, relief)?",
    #     "Does the ad tap into a common desire or aspiration of the target audience?",
    #     "Does the ad create a sense of urgency or scarcity (if appropriate)?",
    #     "Does the ad use storytelling or relatable scenarios to connect with the audience?",
    #     "Does the ad make the audience feel understood or validated?"
    # ],
    "emotional_appeal":[
        "Does the ad tap into a core emotional need or common desire of the target audience?",
        "Does the ad create a clear vision of potential transformation or highlight the emotional benefits of the product/service?",
        "Does the ad make the audience feel understood or validated?",
        "Does the ad evoke a positive emotion (e.g., joy, excitement, relief)?",
        "Does the ad create a sense of urgency or scarcity (if appropriate)?",
        "Does the ad use storytelling or relatable scenarios to connect with the audience?"
    ],
    # "relevance_clarity": [
    #     "Is the core message of the ad immediately clear and easy to understand?",
    #     "Does the script directly highlight the key benefits of the product/service for the target audience?",
    #     "Is the language used simple and free of jargon or overly technical terms?",
    #     "Does the script explicitly address a specific pain point, desire, or need of the target audience as mentioned in the brief?",
    #     "Is it obvious what the viewer is supposed to understand or take away from the ad?",
    #     "Does the script clearly link the product/service to the solution of the identified pain point or the fulfillment of the desire?",
    #     "Is the call to action (if present) unambiguous and easy to follow?",
    #     "Does the script avoid unnecessary complexity or tangential information?",
    #     "Is the intended meaning of each sentence and phrase readily apparent?",
    #     "Does the script maintain focus on the most relevant aspects of the product/service for the target audience?",
    #     "Does the script proactively address potential customer hesitations?",
    #     "Are logical arguments provided to support emotional claims?",
    #     "Does the ad include credibility markers (testimonials, proof points)?",
    #     "Is there clear, transparent information that builds trust?",
    #     "Is the call-to-action clear and straightforward?",
    #     "Are the next steps for purchase or engagement explicitly outlined?",
    #     "Does the ad reduce perceived friction in taking action?",
    #     "Is there a sense of urgency or compelling reason to act now?"
    # ],
    "relevance_clarity":[
        "Is the core message of the ad clear and easy to understand?",
        "Does the script directly highlight the key benefits and maintain focus on the most relevant aspects of the product/service for the target audience?",
        "Does the script explicitly address a specific pain point, desire, or need of the target audience as mentioned in the brief?",
        "Is the call to action clear, unambiguous, and easy to follow?",
        "Does the script avoid unnecessary complexity or tangential information?",
        "Is the intended meaning of each sentence and phrase readily apparent?",
        "Does the script proactively address potential customer hesitations?",
        "Does the ad provide logical arguments to support emotional claims?",
        "Does the ad include credibility markers (testimonials, proof points)?",
        "Are the next steps for purchase or engagement explicitly outlined?",
        "Does the ad reduce perceived friction in taking action?"
    ],
    # "natural_language": [
    #     "Does the ad script sound like something a human would naturally say or write?",
    #     "Is the sentence structure varied and natural-sounding?",
    #     "Does the script use contractions and colloquialisms where appropriate for the target audience and brand voice?",
    #     "Does the language flow smoothly and conversationally?",
    #     "Does the script avoid overly formal, stiff, or robotic phrasing?",
    #     "Are there instances of awkward or unnatural wording in the script?",
    #     "Does the tone of voice feel authentic and consistent throughout the script?",
    #     "Does the script use language that feels relatable and down-to-earth?",
    #     "Does the ad avoid sounding like a template or a collection of keywords?",
    #     "Does the overall language feel human-generated rather than artificially constructed?",
    #     "Does the ad use personal pronouns (you, your) effectively?",
    #     "Is the language conversational and casual?",
    #     "Does the content sound like a dialogue, not a monologue?",
    #     "Are industry-specific jargons minimized?",
    #     "Does the ad use the audience's own language and phrases?"
    # ],
    "natural_language":[
        "Does the language feel natural and human-generated, rather than artificially constructed?",
        "Does the sentence structure vary, with language flowing smoothly and conversationally?",
        "Does the script use contractions and colloquialisms appropriately for the target audience and brand voice?",
        "Does the script avoid overly formal, stiff, or robotic phrasing, and also avoid sounding like a template or collection of keywords?",
        "Is the wording in the script natural and free from awkwardness?",
        "Does the tone of voice feel authentic and consistent throughout the script?",
        "Does the script use relatable, down-to-earth language that feels like the audience's own phrases?",
        "Does the content sound like a dialogue, not a monologue?",
        "Are industry-specific jargons minimized?"
    ],
    # "system1_assessment": [
    #     "Does the ad create an immediate emotional response?",
    #     "Are visual and sensory cues designed for quick, intuitive understanding?",
    #     "Does the ad trigger an instant gut feeling or intuitive reaction?",
    #     "Is the core message comprehensible in less than 3 seconds?"
    # ],
    "system1_assessment":[
    "Does the ad create an immediate emotional reaction in the viewer?",
    "Does the ad create an immediate instinctive reaction in the viewer?",
    "Are visual and sensory cues designed for intuitive understanding, with the core message easily comprehensible within 3 seconds?"
    ],
    # "system2_validation": [
    #     "Does the ad provide logical reasoning behind emotional claims?",
    #     "Are detailed product/service information clearly presented?",
    #     "Does the content address potential rational concerns?",
    #     "Is there a smooth transition between emotional and rational elements?"
    # ],
    "system2_validation":[
    "Does the ad provide logical reasoning behind emotional claims and address potential rational concerns?",
    "Are detailed product/service information clearly presented?",
    "Is there a smooth transition between emotional and rational elements?"
    ],
    # "cognitive_harmony": [
    #     "Does the ad minimize cognitive load?",
    #     "Are complex ideas simplified without losing essential meaning?",
    #     "Does the content enable self-persuasion?",
    #     "Is there a balanced approach between emotional and rational appeal?"
    # ],
    "cognitive_harmony":[
    "Does the ad minimize cognitive load by simplifying complex ideas without losing essential meaning?",
    "Does the content enable self-persuasion?",
    "Is there a balanced approach between emotional and rational appeal?"
    ],
    # "red_flags": [
    #     "Does the ad avoid being overly manipulative?",
    #     "Are claims substantiated and not exaggerated?",
    #     "Does the content feel authentic and not forced?",
    #     "Is the ad free from contradictory messaging?"
    # ],
    "red_flags":[
    "Does the ad avoid being overly manipulative and feel authentic, not forced?",
    "Are claims substantiated and not exaggerated?",
    "Is the ad free from contradictory messaging?"
    ],
    # "hallucination_check": [
    #     "Are all product claims verified in references?",
    #     "Are all features mentioned supported by documentation?",
    #     "Are all benefits stated factual and verifiable?",
    #     "Are all statistics or data points accurate?",
    #     "Are all comparisons supported by references?",
    #     "Are all technical specifications correct?",
    #     "Are all customer testimonials authentic?",
    #     "Are all pricing claims accurate?",
    #     "Are all performance claims verified?",
    #     "Are all brand statements accurate?"
    # ]
    "hallucination_check":[
    "Are all product claims and features verified by references or supported by documentation?",
    "Are all benefits and performance claims factual and verifiable?",
    "Are all statistics or data points accurate?",
    "Are all comparisons supported by references?",
    "Are all technical specifications correct?",
    "Are all customer testimonials authentic?",
    "Are all pricing claims accurate?",
    "Are all brand statements accurate?"
    ]
}

CATEGORY_DISPLAY_NAMES = {
    "creativity": "Creativity",
    "emotional_appeal": "Emotional Appeal",
    "relevance_clarity": "Relevance & Clarity",
    "natural_language": "Natural Language",
    "system1_assessment": "System 1 (Intuitive) Assessment",
    "system2_validation": "System 2 (Analytical) Validation",
    "cognitive_harmony": "Cognitive Processing Harmony",
    "red_flags": "Red Flags and Caution Points",
    "hallucination_check": "Hallucination Check"
}

def get_questions_for_category(category: str) -> list:
    """Get the list of questions for a specific category."""
    return EVALUATION_QUESTIONS.get(category, []) 