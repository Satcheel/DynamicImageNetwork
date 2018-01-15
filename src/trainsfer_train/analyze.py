# 本文件用来生成训练结果的曲线图

import matplotlib.pyplot as plt
import os
import pickle
from matplotlib.backends.backend_pdf import PdfPages
from pprint import pprint
import matplotlib.ticker as mtick

pp = PdfPages('unpre.pdf')
pstatics = []
upstatics = []
for file_name in os.listdir("result"):
    with open("result/"+file_name,"rb") as f:
        s = pickle.load(f)

        if not file_name.find("unpretrain")==-1:
            upstatics.append(s)
        else:
            pstatics.append(s)
        print(f"{file_name} {s['val_accuracy'][-1]}")

statics = upstatics
colors = "brcmgykw"
fig=plt.figure(figsize=(10,10))
# f, ((ax11, ax12), (ax13, ax14)) = plt.subplots(2, 2, sharex=True, sharey=True)
plt.subplots_adjust(left=0.03, bottom=0.03, right=0.93, top=0.95, wspace=0.15, hspace=0.25)
ax11=plt.subplot(221)
for i in range(len(pstatics)):
    plt.plot(statics[i]["train_accuracy"], label=statics[i]["name"], color=colors[i], linestyle="-")
ax11.yaxis.tick_right()
plt.title("Accuracy(train)")
plt.xlabel("epoch")
# plt.ylabel("acc")
plt.legend(loc=4)
ax13=plt.subplot(223)
for i in range(len(pstatics)):
    plt.plot(statics[i]["train_loss"], label=statics[i]["name"], color=colors[i], linestyle="-")
ax13.yaxis.tick_right()
plt.title("Loss(train)")
plt.xlabel("epoch")
# plt.ylabel("loss")
plt.legend(loc=1)

ax12=plt.subplot(222)
for i in range(len(upstatics)):
    plt.plot(statics[i]["val_accuracy"], label=statics[i]["name"], color=colors[i], linestyle="-")
ax12.yaxis.tick_right()
plt.title("Accuracy(validation)")
plt.xlabel("epoch")
# plt.ylabel("acc")
plt.legend(loc=4)
ax14=plt.subplot(224)
for i in range(len(upstatics)):
    plt.plot(statics[i]["val_loss"], label=statics[i]["name"], color=colors[i], linestyle="-")
ax14.yaxis.tick_right()
plt.title("Loss(validation)")
plt.xlabel("epoch")
# plt.ylabel("loss")
plt.legend(loc=1)

plt.show()
pp.savefig(fig)
pp.close()