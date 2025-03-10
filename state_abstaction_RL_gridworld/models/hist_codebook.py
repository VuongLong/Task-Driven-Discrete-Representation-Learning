class HistogramCodebook:
    """
    Measure the historgram of the codebook,
    """

    def __init__(self, codebook_size, running_window: int = 10_000):
        from collections import deque

        # only measure the last `running_window` timestep prior to the present
        self.q = deque(maxlen=running_window)

    def update(self, codebook_indx):
        self.q.extend(codebook_indx)

    def measure(self, tag: str):
        import matplotlib.pyplot as plt

        plt.hist(self.q)
        plt.savefig("hist" + tag + ".pdf")
