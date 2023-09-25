from visualdl.server import app


list1 = ['./train_log/simple_adversary_0_60', './train_log/simple_adversary_1_60',
         './train_log/simple_adversary_2_60', './train_log/simple_adversary_3_60',
         './train_log/simple_adversary_4_60', './train_log/simple_adversary_5_60',
         ]

if __name__ == '__main__':
    app.run(logdir=list1)
