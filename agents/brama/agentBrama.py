import os
import re
import requests
import json
import base64
import io
from PIL import Image
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from checkDMARC import *
from braveSearch import *
from educationalModuleRAG import *

requests.packages.urllib3.disable_warnings()

# API Keys - Only XAI_API_KEY is required for core functionality
BRAVE_API_KEY=os.environ.get('BRAVE_API_KEY')
xai_api_key = os.environ.get('XAI_API_KEY')
voyage_api_key = os.environ.get('VOYAGE_API_KEY')  # Only for educational mode
umbrella_api_client = os.environ.get('UMBRELLA_API_CLIENT')
umbrella_api_secret = os.environ.get('UMBRELLA_API_SECRET')
url_hause_key = os.environ.get('URL_HAUSE_KEY')

# Only XAI_API_KEY is required; all others are optional
if not xai_api_key:
    raise ValueError("XAI_API_KEY is required. Please set XAI_API_KEY environment variable.")


class CybersecurityAgent:
    def __init__(self):
        self.xai_api_key = os.environ.get('XAI_API_KEY')
        self.vt_api_key = os.environ.get('VT_API_KEY')
        self.brave_api_key = os.environ.get('BRAVE_API_KEY')

        # Only XAI_API_KEY is required
        if not self.xai_api_key:
            raise ValueError("XAI_API_KEY is required. Please set XAI_API_KEY environment variable.")

        # xAI uses OpenAI-compatible API, so we use ChatOpenAI with xAI's base URL
        self.llm = ChatOpenAI(
            model="grok-beta",
            api_key=self.xai_api_key,
            base_url="https://api.x.ai/v1"
        )
        self.vision_llm = ChatOpenAI(
            model="grok-beta",
            api_key=self.xai_api_key,
            base_url="https://api.x.ai/v1"
        )
        self.setup_agent()


    def queryUrlHause(self, url):
        if not url.startswith('http://') and not url.startswith('https://'):
            url = 'https://' + url
        
        if not url_hause_key:
            return "Error: URL_HAUSE_KEY environment variable not set."
        
        data = {'url': url}
        headers = {
            'Auth-Key': url_hause_key
        }
        
        response = requests.post('https://urlhaus-api.abuse.ch/v1/url/', headers=headers, data=data)
        json_response = response.json()
        
        if json_response['query_status'] == 'ok':
            return json.dumps(json_response, indent=4, sort_keys=False)
        elif json_response['query_status'] == 'no_results':
            url = 'http://' + url[8:]
            response = requests.post('https://urlhaus-api.abuse.ch/v1/url/', headers=headers, data={'url': url})
            json_response = response.json()
            if json_response['query_status'] == 'ok':
                return json.dumps(json_response, indent=4, sort_keys=False)
            elif json_response['query_status'] == 'no_results':
                return "No results"
            else:
                return "Something went wrong"
        else:
            return "Something went wrong"


    def queryVirusTotal(self, url):
        """Query VirusTotal API if available, otherwise return None."""
        if not self.vt_api_key:
            return "VirusTotal API key not configured. Skipping VirusTotal scan."
        
        try:
            VTapiEndpoint = "https://www.virustotal.com/api/v3/urls"
            payload = f'url={url}'
            headers = {
                'x-apikey': self.vt_api_key,
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            response = requests.post(VTapiEndpoint, headers=headers, data=payload)
            try:
                VTurlID = response.json()["data"]["links"]["self"]
                response = requests.request("GET", VTurlID, headers=headers)
                return response.text
            except KeyError:
                return "Error: Invalid response data from VirusTotal API"
        except Exception as e:
            return f"Error accessing VirusTotal API: {str(e)}"

    def getDomainsRiskScore(self, url):

        api_url = "https://api.umbrella.com/auth/v2/token"

        usrAPIClientSecret = umbrella_api_client + ":" + umbrella_api_secret
        basicUmbrella = base64.b64encode(usrAPIClientSecret.encode()).decode()
        HTTP_Request_header = {"Authorization": "Basic %s" % basicUmbrella,
                                "Content-Type": "application/json;"}

        payload = json.dumps({
        "grant_type": "client_credentials"
        })

        response = requests.request("GET", api_url, headers=HTTP_Request_header, data=payload)


        try:
            accessToken = response.json()['access_token']

        except KeyError:
            return "Error: Invalid response data from Umbrella; check your API credential"
        

        api_url = "https://api.umbrella.com/investigate/v2/domains/risk-score/{}".format(url)

        payload = {}
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + accessToken
        }
        response = requests.request("GET", api_url, headers=headers, data=payload)

        try:
            json_data = response.json()
            return {"domain": url, "risk_score": json_data["risk_score"]}
        except KeyError:
            return "Error in receiving the results: Risk score not found in the response"
        except Exception as e:
            return f"Error in receiving the results: {str(e)}"
    

    def analyze_domain(self, url):
        """Analyze domain using available APIs, with fallback to xAI-only analysis."""
        vt_data = self.queryVirusTotal(url)
        urlhaus_data = self.queryUrlHause(url)
        umbrella_data = self.getDomainsRiskScore(url)
        
        # Check if we have any external scan data
        has_external_data = (
            (vt_data and not vt_data.startswith("Error") and not "not configured" in vt_data) or
            (urlhaus_data and not urlhaus_data.startswith("Error")) or
            (umbrella_data and not isinstance(umbrella_data, str) or not umbrella_data.startswith("Error"))
        )
        
        if has_external_data:
            # Use template with external scan data
            template = """
            Analyze the following JSON data from domain scan sources:
            VirusTotal scan: {JSON_DATA_Virus_Total}
            URLhaus scan: {JSON_DATA_URL_HOUSE}
            Umbrella scan: {JSON_DATA_Umbrella}

            Based on the analysis, generate a brief assessment following these rules:
            1. Start with "Based on related databases, domain identified as [malicious/suspicious/secure]"
            2. Use "malicious" if VirusTotal malicious count > 0 or URLhaus query_status is "ok"
            3. Use "suspicious" if VirusTotal suspicious count > 0 or undetected count is high
            4. Use "secure" if VirusTotal harmless count is high and malicious/suspicious counts are 0, and URLhaus query_status is "no_results"
            5. Highlight the URL status as online/offline/unknown from URLhaus data
            6. Check the blacklists key in URLhaus data and highlight if the domain is identified as a spammer domain, phishing domain, botnet C&C domain, compromised website, or not listed
            7. Check Umbrella scan data. The domain is malicious if the domain risk_score value is close to 100. Domains with risk_score values from 0 to 40 are safe.
            8. Provide a short summary of up to 10 words
            9. Add a brief description if needed, focusing on key findings

            Output the assessment in a concise paragraph.
            """
            prompt = PromptTemplate(template=template, input_variables=["JSON_DATA_Virus_Total", "JSON_DATA_URL_HOUSE", "JSON_DATA_Umbrella"])
            chain = prompt | self.llm | RunnableLambda(lambda x: x.content)
            return chain.invoke({"JSON_DATA_Virus_Total": vt_data, "JSON_DATA_URL_HOUSE": urlhaus_data, "JSON_DATA_Umbrella": umbrella_data})
        else:
            # Fallback: Use xAI to analyze the domain directly
            template = """
            Analyze the following domain/URL for potential security risks: {URL}
            
            Provide a security assessment covering:
            1. Domain reputation and potential threats
            2. Common security concerns (phishing, malware, scams)
            3. Recommendations for users
            
            Output a concise security assessment in 2-3 sentences.
            """
            prompt = PromptTemplate(template=template, input_variables=["URL"])
            chain = prompt | self.llm | RunnableLambda(lambda x: x.content)
            return chain.invoke({"URL": url})

    def describe_image(self, image_path):
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            img_data = io.BytesIO()
            img.save(img_data, format='JPEG')
            img_data.seek(0)
            image_data = base64.b64encode(img_data.read()).decode('utf-8')

        model = ChatOpenAI(
            model="grok-beta",
            api_key=self.xai_api_key,
            base_url="https://api.x.ai/v1"
        )
        IMAGE_DESCRIPTION_PROMPT = """
        Analyze the following image in detail:

        1. Describe the overall layout and visual elements of the image.

        2. Extract and list ALL text visible in the image, exactly as it appears. Do not paraphrase or summarize. Include:
           - Headings
           - Body text
           - Labels
           - Buttons
           - Any other visible text

        3. Identify and list any of the following types of information, if present:
           - Email addresses
           - Phone numbers
           - Web domains
           - IP addresses
           - Social media handles
           - Names of people or organizations
           - Dates
           - Locations

        4. Note any logos, icons, or distinctive visual elements.

        5. Describe any charts, graphs, or data visualizations, if present.

        6. Mention any notable color schemes or design elements.

        7. If the image appears to be a screenshot of a specific type of content (e.g., email, social media post, web page), identify it.

        Please be as thorough and precise as possible in your analysis, ensuring that all text is captured exactly as it appears in the image.
        """
        message = HumanMessage(
            content=[
                {"type": "text", "text": IMAGE_DESCRIPTION_PROMPT},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data,
                    },
                },
            ],
        )
        response = model.invoke([message])
        return response.content

    def setup_agent(self):
        tools = [
            Tool(
                name="Domain Analyzer",
                func=self.analyze_domain,
                description="Analyzes a domain or URL for potential security threats"
            ),
            Tool(
                name="Message Analyzer",
                func=self.analyze_message,
                description="Analyzes a message for phishing attempts"
            ),
            Tool(
                name="Phone Number Analyzer",
                func=self.analyze_phone,
                description="Analyzes a phone number for potential threats"
            )
        ]
        self.agent = initialize_agent(tools, self.llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    def analyze_phone(self, phone_number):
        """Analyze phone number, with fallback to xAI-only analysis if Brave API is not available."""
        phoneNumberVariants = generate_phone_number_variants(phone_number)
        
        # Try to get search data if Brave API is available
        if self.brave_api_key:
            try:
                searchData = checkPhoneLogic(phoneNumberVariants)
                template = """{Search_Data_Brave} \n\nPlease answer the user's question using only information from the search results. Include links to the relevant search result URLs within your answer. Keep your answer concise.

                User's question: Can you identify whether telephone number variants contains in this list {Phone_Number_Variants} is used for scams, phishing, or other suspicious activities? Highlight if the number is unsafe or write that there needs to be more information or if negative reviews and comments were not recorded in the first ten search results sites. 

                Assistant:
                """
                prompt = PromptTemplate(template=template, input_variables=["Search_Data_Brave", "Phone_Number_Variants"])
                chain = prompt | self.llm | RunnableLambda(lambda x: x.content)
                return chain.invoke({"Search_Data_Brave": searchData, "Phone_Number_Variants": phoneNumberVariants})
            except Exception as e:
                # Fall through to xAI-only analysis if Brave search fails
                pass
        
        # Fallback: Use xAI to analyze phone number directly
        template = """Analyze the following phone number for potential security risks: {Phone_Number}

        Phone number variants to check: {Phone_Number_Variants}
        
        Provide a security assessment covering:
        1. Common phone number scams and phishing patterns
        2. Whether this number format matches known suspicious patterns
        3. General recommendations for handling calls/texts from unknown numbers
        
        Note: Without web search capabilities, this is a general assessment based on number patterns.
        Output a concise security assessment in 2-3 sentences.
        """
        prompt = PromptTemplate(template=template, input_variables=["Phone_Number", "Phone_Number_Variants"])
        chain = prompt | self.llm | RunnableLambda(lambda x: x.content)
        return chain.invoke({"Phone_Number": phone_number, "Phone_Number_Variants": ", ".join(phoneNumberVariants)})
    
    

    def analyze_message(self, message):
        template = """
        Analyze the following message for potential phishing attempts:
        Message: {message}

        Provide your analysis, highlighting any suspicious elements:
        If message contains domain or email than also call Domain Analyzer tool, if contains phone number than call Phone Number Analyzer Tool
        """
        prompt = PromptTemplate(template=template, input_variables=["message"])
        chain = prompt | self.llm | RunnableLambda(lambda x: x.content)
        return chain.invoke({"message": message})

    def run(self):
        while True:
            user_input = input("Hi, this is an AI Agent Brama, who can help you check the security metrics and safety of the following resources: \nText messages, Site URL, Email, Phone number, and SMS. You can also use the educational mode to learn more about social engineering and cybersecurity threats, such as scams and phishing.\n\nEnter a URL, message, or write 'img', 'screenshot', or 'image' to attach an image, or 'education_mode' or 'quit' to exit: ")
            if user_input.lower() == 'quit':
                break

            # Check if user wants to attach a text file
            if user_input.lower() == 'file':
                file_path = input("Enter the path to the text file: ")
                with open(file_path, 'r') as file:
                    user_input = file.read()

            elif user_input.lower() == 'education_mode':
                educational_mode()
                continue

            # Extract domain from email address in user input
            email_match = re.search(r'\b[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b', user_input)
            if email_match:
                domain = email_match.group(1)
                domain_analysis = self.analyze_domain(domain)
                dmarc_analysis = checkDMARC(domain)
                user_input += f"\n\nDMARC analysis: {dmarc_analysis}"
            
            # Extract domain from user input
            domain_match = re.search(r'([A-Za-z0-9.-]+\.[A-Z|a-z]{2,})', user_input)
            if domain_match:
                domain = domain_match.group(1)
                if 'Domain Analyzer' in [tool.name for tool in self.agent.tools]:
                    domain_analysis = self.agent.run(f"Analyze the domain: {domain}")
                    user_input += f"\n\nDomain analysis: {domain_analysis}"

            # Extract phone number from user input
            phone_match = re.search(r'\+?\d[\d -]{8,15}\d', user_input)
            
            if phone_match:
                phone_number = phone_match.group(0)
                phone_analysis = self.analyze_phone(phone_number)
                user_input += f"\n\nPhone analysis: {phone_analysis}"

            # Check if user wants to attach a screenshot
            if 'img' in user_input.lower() or 'screenshot' in user_input.lower() or 'image' in user_input.lower():
                image_path = input("Enter the path to the image: ")
                image_analysis = self.describe_image(image_path)
                user_input += f"\n\nImage analysis: {image_analysis}"

            response = self.agent.run(user_input)
            

            # Analyze message content
            message_analysis = self.analyze_message(response)
            print(f"Message analysis: {message_analysis}")

if __name__ == "__main__":
    try:
        agent = CybersecurityAgent()
        agent.run()
    except ValueError as e:
        print(f"Error: {e}")
