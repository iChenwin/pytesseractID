"""
    多线程测试
"""
#coding=utf-8
from time import ctime, sleep
import threading

def music(name):
    for i in range(2):
        print "I'm listening %s at %s" % (name, ctime())
        sleep(1)
        print "music done %s" % ctime()

def movie(name):
    for i in range(2):
        print "I'm watching %s at %s" % (name, ctime())
        sleep(5)
        print "movie done %s" % ctime()

threads = []
t1 = threading.Thread(target = music, args = (u"成都",))
threads.append(t1)
t2 = threading.Thread(target = movie, args = (u"阿凡达",))
threads.append(t2)

if __name__ == "__main__":
    for t in threads:
        t.setDaemon(True)
        t.start()
    t.join()
    print "It's over! %s" % ctime()