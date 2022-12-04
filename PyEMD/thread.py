import threading


class TimedJob(threading.Thread):
    def __init__(self, interval, execute, *args):
        threading.Thread.__init__(self)
        # self.daemon = False
        self.stopped = threading.Event()
        self.interval = interval
        self.execute = execute
        self.args = args

    def stop(self):
        self.stopped.set()
        self.join()

    def run(self):
        while not self.stopped.wait(self.interval.total_seconds()):
            self.execute(*self.args)
