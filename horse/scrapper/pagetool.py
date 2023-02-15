from selenium.webdriver.common.by import By
import re


def get_performance_info(webdriver, race_date, race_no, race_course, base_url):
    from time import time

    t0=time()

    target_url = base_url.format(race_date, race_course, race_no)
    webdriver.get(target_url)
    webdriver.implicitly_wait(1)

    # try to get performance table
    for _ in range(3):
        try:
            performance = webdriver.find_element(by=By.XPATH, value='//*[@id="innerContent"]/div[2]/div[5]')
        except:
            webdriver.implicitly_wait(1)
    
    performance_file = open('./data/performance.txt', 'a', encoding='utf-8')
    horse_file = open('./data/url_horse.txt', 'a', encoding='utf-8')
    jockey_file = open('./data/url_jockey.txt', 'a', encoding='utf-8')
    trainer_file = open('./data/url_trainer.txt', 'a', encoding='utf-8')

    # main body
    try:
        """ race info """
        distance = webdriver.find_element(by=By.XPATH, value='//*[@id="innerContent"]/div[2]/div[4]/table/tbody/tr[2]/td[1]').text
        field_going = webdriver.find_element(by=By.XPATH, value='//*[@id="innerContent"]/div[2]/div[4]/table/tbody/tr[2]/td[3]').text
        race_name = webdriver.find_element(by=By.XPATH, value='//*[@id="innerContent"]/div[2]/div[4]/table/tbody/tr[3]/td[1]').text
        course_type_n_no = webdriver.find_element(by=By.XPATH, value='//*[@id="innerContent"]/div[2]/div[4]/table/tbody/tr[3]/td[3]').text
        race_money = webdriver.find_element(by=By.XPATH, value='//*[@id="innerContent"]/div[2]/div[4]/table/tbody/tr[4]/td[1]').text

        performance_ranks = performance \
            .find_element(by=By.TAG_NAME, value='tbody') \
            .find_elements(by=By.TAG_NAME, value='tr')

        # horse = {}
        # jockey = {}
        # trainer = {}

        # performance_data = []
        """ table """
        for prank in performance_ranks:
            prank_data = prank.find_elements(by=By.TAG_NAME, value='td')
            performance_elem = [race_date, race_no, race_course, distance, field_going, race_name, course_type_n_no, race_money]
            for ix, data in enumerate(prank_data):
                performance_elem.append(data.text)

                href = re.findall(r'href="(.*?)"', data.get_attribute('innerHTML'))
                if len(href)>0:
                    if ix==2:
                        # horse[performance_elem[-1]] = href[0]
                        horse_file.write(f'{performance_elem[-1]}:{href[0]}\n')
                    elif ix==3:
                        # jockey[performance_elem[-1]] = href[0]
                        jockey_file.write(f'{performance_elem[-1]}:{href[0]}\n')
                    elif ix==4:
                        # trainer[performance_elem[-1]] = href[0]
                        trainer_file.write(f'{performance_elem[-1]}:{href[0]}\n')

            # performance_data.append(performance_elem)
            performance_file.write(':'.join([str(i) if i !='' else ' ' for i in performance_elem])+'\n')

    except:
        print('Err')
        performance_file.close()
        horse_file.close()
        jockey_file.close()
        trainer_file.close()
        return False

    performance_file.close()
    horse_file.close()
    jockey_file.close()
    trainer_file.close()

    t1=time()

    print(f'Scrapping: [Cost] - {round(t1-t0, 2)}s. Target - {target_url}')

    return True