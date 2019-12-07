from selenium import webdriver
from selenium.common import exceptions
from selenium.webdriver.common.keys import Keys
import pdb

####
##  Configuration
MSGBOX = "_3u328"
SENDBUTTON = "_3M-N-"
NOTIFICATIONFLAG = "P6z4j"
MESSAGEIN = "message-in"
MESSAGESPAN = "_F7Vk"
##
####

class Whats():
  def __init__(self):
    self.page = webdriver.Firefox()
    self.page.get("https://web.whatsapp.com/")
    input("Digite qualquer coisa depois de scanear o codigo QR")

  def getLastText(self):
    messages = self.page.find_elements_by_class_name(MESSAGEIN)
    lastMessage = messages.pop()
    textSpan = lastMessage.find_element_by_class_name(MESSAGESPAN)

    try:
      text = textSpan.find_element_by_tag_name("span").text
    except exceptions.NoSuchElementException:
      text = ""

    return text

  def waitForNewMessage(self):
    # pdb.set_trace()
    calls = self.page.find_elements_by_class_name(NOTIFICATIONFLAG)

    while(len(calls) == 0):
      calls = self.page.find_elements_by_class_name(NOTIFICATIONFLAG)

    call = calls.pop()

    try:
      call.click()
    except exceptions.StaleElementReferenceException:
      call = None

    text = self.getLastText()
    
    return (call,text)

  def answer(self, call, answer):
    # try:
    #   call.click()
    # except exceptions.StaleElementReferenceException:
    #   return

    msgBox = self.page.find_element_by_class_name(MSGBOX)
    resp = answer.split("\n")
    for r in resp:
      msgBox.send_keys(r)
      msgBox.send_keys(Keys.SHIFT, Keys.ENTER)
    button = self.page.find_element_by_class_name(SENDBUTTON)
    button.click()
