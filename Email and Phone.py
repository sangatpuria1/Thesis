import re
import matplotlib.pyplot as plt
from collections import Counter
from HuggingFacePerson import count


#Regex to look for phone patterns that start with +
def extract_phone_numbers(text):
    phone_pattern = re.compile(r'''
        # Phone numbers must start with +
        \+
        # Followed by digits and optional separators
        (?:
            \d        
            [\s\-()]*    
        ){8,}                      
        \d             
    ''', re.VERBOSE)

    #Looking for potential matches
    potential_numbers = phone_pattern.findall(text)

    #Filter: Keep only those with ≥8 digits after removing non-digits
    valid_numbers = []
    for num in potential_numbers:
        digits = re.sub(r'[^0-9]', '', num)  # Remove non-digits
        if len(digits) >= 8:
            valid_numbers.append(num.strip())  # Keep original format

    return valid_numbers


def extract_emails(text):
    #Extracting E-mail addresses from the text
    email_pattern = re.compile(r'''
        [a-zA-Z0-9._%+-]+  # username part
        @                  # @ symbol
        [a-zA-Z0-9.-]+     # domain name
        \.                 # dot
        [a-zA-Z]{2,}       # top-level domain
    ''', re.VERBOSE)

    #Remove spaces from found emails (they might appear due to re.VERBOSE)
    emails = [email.replace(" ", "") for email in email_pattern.findall(text)]
    return emails

#Creating barchart for the most common emails
def create_email_barchart(emails):
    if not emails:
        print("\nNo email addresses found to create bar chart.")
        return

    #Count full email occurrences
    email_counts = Counter(emails)

    #Get top 10 most common full email addresses
    top_emails = email_counts.most_common(10)

    #Prepare data for plotting
    email_addresses, counts = zip(*top_emails)

    #Create bar chart
    plt.figure(figsize=(12, 6))
    bars = plt.bar(email_addresses, counts, color='skyblue')

    #Add counts on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}',
                 ha='center', va='bottom')

    #Customize chart
    plt.title('Top 10 Most Common Email Addresses')
    plt.xlabel('Email Address')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    #Save and show the chart
    plt.savefig('top_emails_barchart.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\nBar chart saved as 'top_emails_barchart.png'")


#Reads the file
file_path = 'combined_content.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

#Extracting and printing the results for phone numbers
print("Phone Numbers Found (starting with + and ≥8 digits):")
phones = extract_phone_numbers(content)
for phone in phones:
    print(phone)
    print(count(phone))

#Extracting and printing the results for email addresses
print("\nEmail Addresses Found:")
emails = extract_emails(content)
for email in emails:
    print(email)
    print(count(email))

#Create bar chart from email addresses
create_email_barchart(emails)