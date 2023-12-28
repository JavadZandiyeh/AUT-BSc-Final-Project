import json
import csv
import scrapy


class MoviesSpider(scrapy.Spider):
    name = "movies"
    allowed_domains = ["imdb.com"]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.1234.5678 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    }

    file_path = 'links.csv'
    with open(file_path, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # headers
        urls = [f"https://www.imdb.com/title/tt{row[1]}/" for row in csv_reader]

    def start_requests(self):
        for url in self.urls:
            yield scrapy.Request(url=url, callback=self.parse, headers=self.headers)

    def parse(self, response):
        data = json.loads(str(response.xpath('/html/head/script[3]/text()').get()))
        popularity = response.xpath('//div[@class="sc-5f7fb5b4-1 bhuIgW"]/text()').get()
        user_reviews = response.xpath('//span[text()="User reviews"]/preceding-sibling::span/text()').get()
        critic_reviews = response.xpath('//span[text()="Critic reviews"]/preceding-sibling::span/text()').get()
        meta_score = response.xpath('//span[text()="Metascore"]/preceding-sibling::span/span/text()').get()

        yield {
            'imdbId': data.get('url').split('/')[-2][2:],
            'type': data.get('@type'),
            'name': data.get('name'),
            'ratingCount': data.get('aggregateRating', {"ratingCount": None}).get('ratingCount'),
            'bestRating': data.get('aggregateRating', {"aggregateRating": None}).get('bestRating'),
            'worstRating': data.get('aggregateRating', {"aggregateRating": None}).get('worstRating'),
            'ratingValue': data.get('aggregateRating', {"aggregateRating": None}).get('ratingValue'),
            'contentRating': data.get('contentRating'),
            'genre': data.get('genre'),
            'datePublished': data.get('datePublished'),
            'keywords': data.get('keywords', "").split(','),
            'actor': [actor['name'] for actor in list(data.get('actor', []))],
            'director': [director['name'] for director in list(data.get('director', []))],
            'creator': [creator['name'] for creator in list(data.get('creator', [])) if creator.get('@type') == 'Person'],
            'duration': data.get('duration'),
            'popularity': popularity,
            'userReviews': user_reviews,
            'criticReviews': critic_reviews,
            'metaScore': meta_score
        }
