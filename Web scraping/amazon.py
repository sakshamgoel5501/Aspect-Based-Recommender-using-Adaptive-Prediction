# importing libraries
from bs4 import BeautifulSoup
import requests
import csv

def main(URL,id):
	# opening our output file in append mode
	File = open("out.csv", "a")

	# specifying user agent, You can use other user agents
	# available on the internet
	HEADERS = ({'User-Agent':
                'Mozilla/5.0 (X11; Linux x86_64)AppleWebKit/537.36 (KHTML, like Gecko)Chrome/44.0.2403.157 Safari/537.36',
                                'Accept-Language': 'en-US, en;q=0.5'})

	# Making the HTTP Request
	webpage = requests.get(URL, headers=HEADERS)

	# Creating the Soup Object containing all data
	soup = BeautifulSoup(webpage.content, "lxml")
	File.write(f"{id}|")
	# retrieving product title
	try:
		# Outer Tag Object
		title = soup.find("h1",attrs={'class':"_2IIDsE _3I-nQy" })

		# Inner NavigableString Object
		title_value = title.string

		# Title as a string value
		title_string = title_value.strip().replace(',', '')

	except AttributeError:
		title_string = "NA"
	print("product Title = ", title_string)

	# saving the title in the file
	File.write(f"{title_string}|")


	# retrieving movie average rating
	
	try:
		
		rating=soup.find("strong", attrs={'class': '_1Q9M0z'}).string.strip().replace(',', '')
	   	    
	except AttributeError:

		rating="NA"
	print("rating = ", rating[0:3])
	rating=rating[0:3]
	File.write(f"{rating}|")
	
	# retrieving movie genres
	try:
		genre = soup.find("div", attrs={'class': '_2KBC2m'})
		genre1=genre.find("dt",text="Genres").findNext("dd").get_text()
	   	    
	except AttributeError:

		genre1="NA"
	print("Genres = ", genre1)

	File.write(f"{genre1}|")
	
	# retrieving movie details
	
	try:
		
	       deatils=soup.find("div", attrs={'dir': 'auto'}).string.strip().replace(',', '')
	   	    
	except AttributeError:

		deatils="NA"
	print("deatils = ", deatils)
	
	File.write(f"{deatils}\n")


	# closing the file
	#File.close()


if __name__ == '__main__':
# opening our url file to access URLs
	filename = open('amazon_dataset.csv', 'r')
	file = csv.DictReader(filename)
	asin=[]
	
	for col in file:
	    asin.append(col['asin'])
	asin = list(set(asin))
	asin = list(filter(None, asin))
	#file = open("url.txt", "r")

	# iterating over the urls
	for id in asin:
	    url="https://www.amazon.com/dp/"+id
	    main(url,id)

