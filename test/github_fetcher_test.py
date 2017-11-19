import unittest
from typing import List

from src import config
from src.github_fetcher import GithubFetcher, extract_php


class GithubFetcherTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

        self.github_fetcher = GithubFetcher(
            config.download_location,
            0,
            150,
            config.proxies,
            config.github_username,
            config.github_password)

    def extract_php_test(self):
        code = "<html><%php something %>other things<%php another thing %>".encode("utf-8")
        php = extract_php(code)
        print(php)
        self.assertEqual(php, " something \n another thing ")


if __name__ == "__main__":
    unittest.main()
