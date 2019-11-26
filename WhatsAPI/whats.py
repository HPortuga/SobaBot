from selenium import webdriver

driver = webdriver.Firefox()
driver.get("https://web.whatsapp.com/")

name = input("Digitar nome de usuario ou grupo : ")
msg = input("Digite sua mensagem : ")
count = int(input("Digite o count : "))

input("Digite qualuqer coisa depois de scanear o codigo QR")

user = driver.find_element_by_xpath('//span[@title = "{}"]'.format(name))
user.click()

msgBox = driver.find_element_by_class_name("_3u328")

for i in range(count):
  msgBox.send_keys(msg)
  button = driver.find_element_by_class_name("_3M-N-")
  button.click()
