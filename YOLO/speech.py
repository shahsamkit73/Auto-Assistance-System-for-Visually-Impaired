import pyttsx3

def my_speak_cloud(my_message):
    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate)
    engine.say('{}'.format(my_message))
    engine.runAndWait()
    #rate = engine.getProperty('rate')

message='''

Hello everyone Welcome to Auto assistance system based on convolutional neural network
My name is Jayraj Bandariya 
i did my Bachelor in electronic engineering 


'''
my_speak_cloud(message)