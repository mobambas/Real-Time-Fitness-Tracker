from google import genai
from google.genai import types
import os 
from dotenv import load_dotenv
import re


# Load environment variables from .env file
load_dotenv() 

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Store conversation history
conversation_history = []

# System instruction for fitness context
SYSTEM_INSTRUCTION = """
You are a fitness expert chatbot designed to assist users with questions about exercise, workouts, and fitness form. Follow these rules:
1. Provide accurate, concise, and practical fitness advice.
2. Focus exclusively on fitness-related topics (e.g., exercise form, workout plans, recovery tips).
3. If a question is outside fitness, politely redirect to fitness (e.g., "I'm here to help with fitness! Could you ask about workouts or exercise form?").
4. For questions about push-ups, sit-ups, squats, or lunges, reference the tracking logic: push-ups (elbow angle <100° for down, back >160°), sit-ups (hip angle <170°, shoulder/nose movement), squats (knee angle <120°), lunges (knee angle <100° per leg).
5. Format responses clearly, using numbered lists for multiple points (e.g., "1. First point.\n2. Second point.").
6. Keep responses under 200 words for brevity.
7. Use an encouraging, energetic tone to motivate users.
"""

def clean_response(text):
    """
    Clean up the response to ensure proper formatting for numbered lists, bullet points, and spacing.
    """
    # Convert (1), (2), etc., to 1., 2., etc.
    text = re.sub(r'\((\d+)\)', r'\1.', text)

    # Remove unnecessary markdown symbols like ** and *
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)

    # Add new lines before numbered points
    text = re.sub(r'(\d+\.\s)', r'\n\1', text)

    # Add new lines after sentences
    text = text.replace('. ', '.\n\n')

    # Add new lines after colons
    text = text.replace(': ', ':\n\n')

    # Ensure paragraphs have space between them
    text = re.sub(r"(\.\s)", r".\n\n", text)

    # Replace * with proper bullet points
    text = text.replace("* ", "\n- ").replace("*", "")

    # Remove excessive new lines
    text = re.sub(r'(\n\s*)+', r'\n', text)

    # Ensure proper spacing for sections like "Workout Structure" or "Important Tips"
    text = re.sub(r'(Workout Structure|Important Tips):', r'\n\1:\n', text)

    # Add spacing before bullet points for clarity
    text = re.sub(r'-\s', r'\n- ', text)

    return text.strip()



def chat_with_fitness_bot(question):
    """Handle user query and return fitness-focused response."""
    global conversation_history
    
    # Append user question to history
    conversation_history.append(types.Content(role="user", parts=[types.Part.from_text(text=question)]))
    
    try:
        # Generate response using Gemini
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=conversation_history,
            config= types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION, temperature=0.7, maxOutputTokens=200)
        )
        
        # Check for valid response
        if not response or not response.candidates:
            return "Error: No response from the fitness bot. Try again!"
        
        # Extract and clean response
        answer = clean_response(response.candidates[0].content.parts[0].text)
        
        # Append AI response to history
        conversation_history.append(types.Content(role="assistant", parts=[types.Part.from_text(text=answer)]))
        
        return answer
    
    except Exception as e:
        return f"Oops! Something went wrong: {str(e)}. Ask me another fitness question!"

def main():
    """Run the fitness chatbot in a terminal interface."""
    print("💪 Welcome to the Fitness Chatbot! Ask about workouts, form, or fitness tips. Type 'exit' to quit. 🚀")
    
    while True:
        user_input = input("\nYour question: ").strip()
        
        if user_input.lower() == 'exit':
            print("Thanks for chatting! Keep crushing those workouts! 🏋️‍♂️")
            break
        
        if not user_input:
            print("Please ask a fitness-related question!")
            continue
        
        response = chat_with_fitness_bot(user_input)
        print(f"\nFitness Bot: {response}")

if __name__ == "__main__":
    main()
