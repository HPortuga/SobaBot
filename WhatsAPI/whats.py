from selenium import webdriver

def waitForNewMessages():
  calls = webPage.find_elements_by_class_name(NOTIFICATIONFLAG)

  while(len(calls) == 0):
    calls = webPage.find_elements_by_class_name(NOTIFICATIONFLAG)

  return calls

####
##  Configuration
MSGBOX = "_3u328"
SENDBUTTON = "_3M-N-"
NOTIFICATIONFLAG = "P6z4j"
##
####

webPage = webdriver.Firefox()
webPage.get("https://web.whatsapp.com/")

input("Digite qualuqer coisa depois de scanear o codigo QR")

calls = waitForNewMessages()
print("Atendendo...")






# name = input("Digitar nome de usuario ou grupo : ")
# msg = input("Digite sua mensagem : ")
# count = int(input("Digite o count : "))

# input("Digite qualuqer coisa depois de scanear o codigo QR")

# user = driver.find_element_by_xpath('//span[@title = "{}"]'.format(name))
# user.click()

# msgBox = driver.find_element_by_class_name("_3u328")

# for i in range(count):
#   msgBox.send_keys(msg)
#   button = driver.find_element_by_class_name("_3M-N-")
#   button.click()
