import requests as re
from bs4 import BeautifulSoup
import os

URL="https://www.udemy.com"
resposne = re.get(URL)

print("response -->" , resposne ,"\n type -->" ,type(resposne))

print("text -->" , resposne.text,"\ncontent -->",resposne.content ,"\nstatus_code -->" , resposne.status_code)

if resposne.status_code != 200:
    print("HTTP connection is not successful! Try again")
else:
    print("HTTP connection is successful")

soup = BeautifulSoup(resposne.content , 'html.parser')

print("title with tags -->" , soup.title, "\n title without tags-->" ,soup.title.text)

for link in soup.find_all("link"):
    print(link.get("href"))

print(soup.get_text())



#  create a folder to save html files
folder = "mini_dataset"

if not os.path.exists(folder):
    os.mkdir(folder)


# 2 A FUNCTION THAT SCRAPES AND RETURNS IT
def scrape_content(URL):
    response = re.get(URL)
    if response.status_code == 200:
        print("HTTP connection is successful! for the URL:", URL)
        return response
    else:
        print("HTTP connection is NOT successful! for the URL:", URL)
        return None


# 3 A FUNCTION TO SAVE A HTML FILE OF THE SCRAPED WEBPAGE IN A DIRECTORY

path = os.getcwd() + "/" + folder  # You can define the path manually.


def save_html(to_where, text, name):
    file_name = name + ".html"
    with open(os.path.join(to_where, file_name), "w", encoding="utf-8") as f:
        f.write(text)

test_text = resposne.text
save_html(path,test_text,"2")

#
# URL_list = [
#     "https://www.kaggle.com",
#     "https://stackoverflow.com",
#     "https://www.researchgate.net",
#     "https://www.python.org",
#     "https://www.w3schools.com",
#     "https://wwwen.uni.lu",
#     "https://github.com",
#     "https://scholar.google.com",
#     "https://www.mendeley.com",
#     "https://www.overleaf.com"
# ]
#
#
# # 5 DEFINE A FUNCTION WHICH TAKES THE URL LIST and RUN STEP 2 and STEP 3 for EACH URL
# def create_mini_dataset(to_where, URL_list):
#     for i in range(0, len(URL_list)):
#         content = scrape_content(URL_list[i])
#         if content is not None:
#             save_html(to_where, content.text, str(i))
#         else:
#             pass
#     print("Mini dataset is created!")
#
#
# create_mini_dataset(path, URL_list)