import streamlit as st
import openai
import requests
import matplotlib.pyplot as plt
import json
from datetime import datetime
import numpy as np

# Set your API keys
openai.api_key = "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"
weather_api_key = "74e04b0ffe2c41d99d251314241608"
news_api_key = "B21b5b0c69c942a7b7f7ec2c7eed7623"
bing_search_api_key = "9b35bdf1013648ea9f2e3e03324bbef3"

# Language support dictionary
language_dict = {
    'English': 'en',
    'French': 'fr',
    'Spanish': 'es',
    'German': 'de',
    'Chinese': 'zh'
}

# Function to fetch news based on city and language
def fetch_news(city, language='en'):
    url = f"https://newsapi.org/v2/everything?q={city}&language={language}&apiKey={news_api_key}"
    response = requests.get(url)
    news_data = response.json()
    
    if news_data['status'] == 'ok':
        articles = news_data['articles']
        news_list = []
        for article in articles[:5]:  # Limiting to top 5 articles
            news_list.append({
                'title': article['title'],
                'description': article['description'],
                'url': article['url']
            })
        return news_list
    else:
        st.error("Failed to fetch news.")
        return []

# Function to display news articles in Streamlit
def display_news(news_list):
    st.subheader("Latest News")
    for news in news_list:
        st.markdown(f"**[{news['title']}]({news['url']})**")
        st.write(news['description'])

# Function to search using Bing API
def bing_search(query, language='en'):
    url = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": bing_search_api_key}
    params = {"q": query, "count": 10, "textDecorations": True, "textFormat": "HTML", "setLang": language}
    
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        search_results = response.json()
        results_list = []
        for result in search_results.get('webPages', {}).get('value', []):
            results_list.append({
                'name': result['name'],
                'snippet': result['snippet'],
                'url': result['url']
            })
        return results_list
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
        return []

# Function to display Bing Search results
def display_bing_results(results):
    st.subheader("Bing Search Results")
    if results:
        for result in results:
            st.markdown(f"**[{result['name']}]({result['url']})**")
            st.write(result['snippet'])
    else:
        st.write("No results found.")

# Mock function for document retrieval
def retrieve_relevant_documents(query):
    documents = [
        "Document 1: Energy-saving tips for cold climates...",
        "Document 2: How to improve insulation in older homes...",
        "Document 3: Energy-efficient appliances and their impact..."
    ]
    return documents

# Function to generate energy-saving recommendations using RAG with a fine-tuned LLM
def generate_energy_saving_recommendations_with_rag(temperature, weather_description, house_size, preference):
    query = f"Temperature: {temperature}°C, Weather: {weather_description}, House Size: {house_size}, Preference: {preference}"
    
    retrieved_docs = retrieve_relevant_documents(query)
    context = "\n".join(retrieved_docs)
    
    prompt = (
        f"Based on the following context:\n{context}\n\n"
        f"The current temperature is {temperature}°C with {weather_description}. "
        f"The house size is {house_size} square feet. The user's priority is {preference}. "
        "Provide detailed energy-saving recommendations, including potential cost savings, environmental impact, and comfort improvements."
    )
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an energy efficiency expert."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.7
    )
    
    return response['choices'][0]['message']['content'].strip()

# Function to get current temperature and weather description from WeatherAPI
def get_current_temperature(api_key, city="Toronto"):
    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}&aqi=no"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        temperature = data['current']['temp_c']
        weather_description = data['current']['condition']['text']
        return temperature, weather_description
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
        return None, None

# Function to visualize potential energy savings
def plot_energy_savings(savings_data):
    st.subheader("Potential Energy Savings")
    fig, ax = plt.subplots()
    ax.bar(savings_data['strategy'], savings_data['savings'])
    st.pyplot(fig)

# Function to predict and plot energy efficiency improvements over time
def plot_efficiency_increase():
    months = np.arange(1, 13)
    efficiency_gain = np.cumsum(np.random.uniform(2, 5, size=12))  # Simulate efficiency gains

    plt.figure(figsize=(10, 5))
    plt.plot(months, efficiency_gain, marker='o', color='b')
    plt.title('Predicted Energy Efficiency Increase Over 12 Months')
    plt.xlabel('Month')
    plt.ylabel('Efficiency Increase (%)')
    plt.grid(True)
    plt.xticks(months)
    st.pyplot(plt)

# Function to calculate the reduction in carbon footprint
def calculate_carbon_footprint_reduction(efficiency_gain):
    initial_carbon_footprint = 100  # Example initial footprint in arbitrary units
    reduced_carbon_footprint = initial_carbon_footprint * (1 - efficiency_gain / 100)
    return reduced_carbon_footprint

# Function to plot carbon footprint reduction
def plot_carbon_footprint(efficiency_gain):
    initial_carbon_footprint = 100  # Example initial footprint in arbitrary units
    months = np.arange(1, 13)
    carbon_footprint = initial_carbon_footprint * (1 - efficiency_gain / 100)

    plt.figure(figsize=(10, 5))
    plt.plot(months, carbon_footprint, marker='o', color='g')
    plt.title('Predicted Carbon Footprint Reduction Over 12 Months')
    plt.xlabel('Month')
    plt.ylabel('Carbon Footprint (arbitrary units)')
    plt.grid(True)
    plt.xticks(months)
    st.pyplot(plt)

def save_or_update_user_profile(user_data, feedback, filename="community_tips.json"):
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tip": user_data.get('tip', 'No tip provided')  # Save the tip directly
    }
    
    try:
        with open(filename, 'r+') as file:
            try:
                profiles = json.load(file)
            except json.JSONDecodeError:
                profiles = []
                
            profiles.append(entry)  # Always append the new tip
            
            file.seek(0)
            json.dump(profiles, file, indent=4)
            file.truncate()

    except FileNotFoundError:
        with open(filename, 'w') as file:
            json.dump([entry], file, indent=4)

    st.write('Community tips have been updated! Please refresh the page to see the changes.')

# Function to load historical recommendations and feedback
def load_historical_data(filename="user_profiles.json"):
    try:
        with open(filename, 'r') as file:
            profiles = json.load(file)
            return profiles
    except (FileNotFoundError, json.JSONDecodeError):
        return []

# Function to check if efficiency goals are met and send alerts
def check_efficiency_goals(efficiency_gain, goal):
    if efficiency_gain[-1] >= goal:
        st.success(f"Congratulations! You've met your efficiency goal of {goal}%!")
    else:
        st.warning(f"Keep going! You're on your way to meeting your efficiency goal of {goal}%.")

def community_sharing_platform():
    st.subheader("Community Energy-Saving Tips")
    tips = load_historical_data("community_tips.json")
    
    if tips:
        for entry in tips:
            timestamp = entry.get('timestamp', 'N/A')
            tip = entry.get('tip', 'No tip provided')  # Correctly access the 'tip' field directly
            st.write(f"**{timestamp}:** {tip}")
            st.write("----")
    else:
        st.write("No community tips available.")
    
    new_tip = st.text_area("Share your energy-saving tip:")
    if st.button("Submit Tip"):
        if new_tip:
            new_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tip": new_tip  # Save the tip directly in the JSON file
            }
            save_or_update_user_profile(new_entry, feedback=None, filename="community_tips.json")
            st.success("Tip submitted successfully! Please refresh the page to see the updated tips.")
        else:
            st.warning("Please enter a tip before submitting.")

# Expert system recommendations
def get_recommendations(city, house_size, insulation, appliances_condition, preference, climate, energy_consumption, renewable_energy, incentives):
    recommendations = []

    # Rule 1: Recommendations based on insulation quality
    if insulation == "poor":
        recommendations.append("Improve insulation to reduce heating and cooling costs.")
        if climate == "cold":
            recommendations.append("Given the cold climate, consider upgrading to high-performance insulation materials to retain heat.")
        elif climate == "hot":
            recommendations.append("In a hot climate, consider radiant barriers and reflective insulation to keep your house cooler.")
    else:
        recommendations.append("Insulation is adequate; ensure it is maintained.")

    # Rule 2: Recommendations based on appliance condition
    if appliances_condition == "old":
        recommendations.append("Consider upgrading to energy-efficient appliances to save energy.")
        if energy_consumption == "high":
            recommendations.append("Since your energy consumption is high, upgrading to ENERGY STAR-rated appliances can significantly reduce your bills.")
    else:
        recommendations.append("Appliances are new; ensure they are regularly maintained.")

    # Rule 3: Recommendations based on house size
    if house_size > 2000:
        recommendations.append("Consider zone heating and cooling to reduce energy costs in larger homes.")
        if energy_consumption == "high":
            recommendations.append("In a large home with high energy consumption, smart thermostats for each zone can help manage energy use efficiently.")
    else:
        recommendations.append("Use programmable thermostats to optimize heating and cooling.")

    # Rule 4: Recommendations based on user preference
    if preference == "cost-saving":
        recommendations.append("Use energy-saving light bulbs and reduce thermostat settings in winter.")
        if incentives:
            recommendations.append("Check for local government incentives on energy-efficient appliances and home upgrades to save costs.")
    elif preference == "environmental":
        recommendations.append("Install solar panels or use green energy providers.")
        if renewable_energy == "solar":
            recommendations.append("Since you already use solar energy, consider adding battery storage to maximize the use of generated energy.")
        elif renewable_energy == "none":
            recommendations.append("Consider investing in renewable energy sources like solar or wind to reduce your carbon footprint.")
    elif preference == "comfort":
        recommendations.append("Use smart thermostats to maintain a comfortable temperature efficiently.")
        if insulation == "poor":
            recommendations.append("Improving insulation can enhance comfort by maintaining a more stable indoor temperature.")

    # Rule 5: Recommendations based on climate
    if climate == "cold":
        recommendations.append("In cold climates, ensure windows and doors are properly sealed to prevent heat loss.")
    elif climate == "hot":
        recommendations.append("In hot climates, consider installing window films or shades to reduce solar heat gain.")

    # Rule 6: Recommendations based on energy consumption habits
    if energy_consumption == "high":
        recommendations.append("Since your energy consumption is high, consider conducting an energy audit to identify areas for improvement.")
        recommendations.append("Implementing energy-efficient practices, like unplugging devices when not in use, can help reduce your energy bills.")
    elif energy_consumption == "low":
        recommendations.append("Your energy consumption is already low, but you can further reduce it by using energy-efficient lighting and appliances.")

    # Rule 7: Recommendations based on renewable energy sources
    if renewable_energy == "solar":
        recommendations.append("You are already using solar energy; consider expanding your solar array or adding a solar water heater to further reduce energy costs.")
    elif renewable_energy == "wind":
        recommendations.append("If wind energy is available in your area, consider adding a small wind turbine to complement your energy needs.")

    # Rule 8: Recommendations based on local government incentives
    if incentives:
        recommendations.append("Take advantage of local government incentives for energy-efficient home improvements and renewable energy installations.")

    return recommendations

# Main function to run the Streamlit app
def main():
    st.title("Energy Efficiency and Information Portal")

    # Language Selection
    language = st.selectbox("Select Language", list(language_dict.keys()))
    selected_language = language_dict[language]

    # User inputs
    city = st.text_input("Enter your city:", "Toronto")
    house_size = st.number_input("Enter the size of your house (in square feet):", min_value=100.0, step=50.0)
    insulation = st.selectbox("Insulation Quality:", ["good", "poor"])
    appliances_condition = st.selectbox("Condition of Appliances:", ["new", "old"])
    preference = st.selectbox("What's your priority?", ("cost-saving", "environmental", "comfort"))
    climate = st.selectbox("Local Climate:", ["cold", "hot", "moderate"])
    energy_consumption = st.selectbox("Energy Consumption Habits:", ["low", "medium", "high"])
    renewable_energy = st.selectbox("Renewable Energy Source:", ["none", "solar", "wind"])
    incentives = st.checkbox("Are there local government incentives available?")

    # Fetch and display news for the selected city and language
    news_list = fetch_news(city, selected_language)
    display_news(news_list)

    # Optional Bing Search
    st.subheader("Optional Search")
    search_query = st.text_input("Enter your search query:")
    if st.button("Search"):
        if search_query:
            search_results = bing_search(search_query, selected_language)
            display_bing_results(search_results)
        else:
            st.warning("Please enter a query to search.")

    # Set Efficiency Goal
    efficiency_goal = st.number_input("Set your efficiency goal (% increase):", min_value=5, max_value=50, step=5, value=20)

    if st.button("Get Recommendations"):
        temperature, weather_description = get_current_temperature(weather_api_key, city)
        
        if temperature is not None:
            st.write(f"**Current temperature in {city}:** {temperature}°C")
            st.write(f"**Weather:** {weather_description}")
            
            # Get Rule-based recommendations
            rule_based_recommendations = get_recommendations(
                city=city, 
                house_size=house_size, 
                insulation=insulation, 
                appliances_condition=appliances_condition, 
                preference=preference, 
                climate=climate, 
                energy_consumption=energy_consumption, 
                renewable_energy=renewable_energy, 
                incentives=incentives
            )
            st.subheader("Rule-based Energy Efficiency Recommendations:")
            for rec in rule_based_recommendations:
                st.write(f"- {rec}")
            
            # Get AI-based recommendations using RAG
            ai_recommendations = generate_energy_saving_recommendations_with_rag(temperature, weather_description, house_size, preference)
            st.subheader("AI-based Energy Efficiency Recommendations:")
            st.write(ai_recommendations)
            
            # Visualization (Example data)
            savings_data = {
                'strategy': ['Insulation Upgrade', 'Appliance Upgrade', 'Zone Heating'],
                'savings': [15, 20, 25]  # hypothetical savings percentages
            }
            plot_energy_savings(savings_data)
            
            # Predict and plot efficiency increase
            st.subheader("Predicted Energy Efficiency Increase")
            plot_efficiency_increase()

            # Predict and plot carbon footprint reduction
            efficiency_gain = np.cumsum(np.random.uniform(2, 5, size=12))  # Simulate efficiency gains
            st.subheader("Predicted Carbon Footprint Reduction")
            plot_carbon_footprint(efficiency_gain)

            # Check if efficiency goals are met and send alerts
            check_efficiency_goals(efficiency_gain, efficiency_goal)
            
            # Feedback system
            feedback = st.slider("Rate the recommendations from 1 to 5", 1, 5, 3)
            if st.button("Submit Feedback"):
                user_data = {
                    "city": city,
                    "house_size": house_size,
                    "insulation": insulation,
                    "appliances_condition": appliances_condition,
                    "preference": preference,
                    "recommendations": {
                        "AI-based": ai_recommendations,
                        "Rule-based": rule_based_recommendations
                    }
                }
                save_or_update_user_profile(user_data, feedback)
                st.success("Feedback submitted successfully!")
        
        else:
            st.error("Failed to retrieve temperature and weather information.")
    
    # Display historical recommendations and feedback
    if st.checkbox("Show Historical Recommendations and Feedback"):
        historical_data = load_historical_data()
        if historical_data:
            st.subheader("Historical Data")
            for entry in historical_data:
                user_data = entry.get('user_data', {})  # Fallback to an empty dict if 'user_data' is missing
                timestamp = entry.get('timestamp', 'N/A')  # Fallback to 'N/A' if timestamp is missing
                st.write(f"**Date/Time:** {timestamp}")
                st.write(f"**City:** {user_data.get('city', 'N/A')}")
                st.write(f"**House Size:** {user_data.get('house_size', 'N/A')} square feet")
                st.write(f"**Insulation Quality:** {user_data.get('insulation', 'N/A')}")
                st.write(f"**Appliances Condition:** {user_data.get('appliances_condition', 'N/A')}")
                st.write(f"**Preference:** {user_data.get('preference', 'N/A')}")
                recommendations = user_data.get('recommendations', {})
                st.write(f"**AI-based Recommendations:** {recommendations.get('AI-based', 'N/A')}")
                st.write(f"**Rule-based Recommendations:** {recommendations.get('Rule-based', 'N/A')}")
                st.write(f"**User Feedback:** {entry.get('feedback', 'N/A')} / 5")
                st.write("----")
        else:
            st.write("No historical data available.")
    
    # Community Sharing Platform
    community_sharing_platform()

if __name__ == "__main__":
    main()


