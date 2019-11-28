from selenium import webdriver
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

  def waitForNewMessages(self):
    calls = self.page.find_elements_by_class_name(NOTIFICATIONFLAG)

    while(len(calls) == 0):
      calls = self.page.find_elements_by_class_name(NOTIFICATIONFLAG)

    return calls

  def getLastText(self):
    messages = self.page.find_elements_by_class_name(MESSAGEIN)
    lastMessage = messages.pop()
    textSpan = lastMessage.find_element_by_class_name(MESSAGESPAN)
    text = textSpan.find_element_by_tag_name("span").text

    return text

  def run(self):
    calls = self.waitForNewMessages()
    for call in calls:
      call.click()
      text = self.getLastText()
      return (call,text)

  def answer(self, call, answer):
    call.click()
    msgBox = self.page.find_element_by_class_name(MSGBOX)
    msgBox.send_keys(answer)
    button = self.page.find_element_by_class_name(SENDBUTTON)
    button.click()
