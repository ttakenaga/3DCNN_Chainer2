# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 18:40:14 2016
入力画像　球体と直方体
3D-CNNで球体と直方体を２クラス分類しましょう！
@author: takenaga
"""

import numpy as np
import cupy as cp
from chainer import cuda, dataset, Chain, initializers, optimizers
from chainer import iterators, training, serializers
import chainer.functions as F
import chainer.links as L
from scipy import ndimage
import argparse
import functools


def rotM(a, b, c):
    # 回転行列を計算する
    # 物体座標系の 1->2->3 軸で回転させる
    Rz = np.array([[1, 0, 0],
                   [0, np.cos(a), np.sin(a)],
                   [0, -np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)],
                   [0, 1, 0],
                   [-np.sin(b), 0, np.cos(b)]])
    Rx = np.array([[np.cos(c), np.sin(c), 0],
                   [-np.sin(c), np.cos(c), 0],
                   [0, 0, 1]])
    R = Rz.dot(Ry).dot(Rx)

    return R


def rotM_c(Euler, coord):
    a, b, c = Euler
    x, y, z = coord
    return np.dot(rotM(a, b, c), np.array((x, y, z)).T)


def MakeImage(shape, ball, cuboid, r_min=2, r_max=15,
              dtype=np.float32, label_dtype=np.int32):
    """
    shape:出力画像のshape
    ball:球形の数
    cuboid:直方体の数
    r_min:（球）最小半径/（直方体）最小重心から頂点までの最小距離
    r_max:（球）最大半径/（直方体）最大重心から頂点までの最大距離
    dtype:出力画像の型
    label_dtype:ラベルの型
    """
    output = np.zeros(shape, dtype=dtype)
    label = np.zeros(shape, dtype=label_dtype)
    distance = np.zeros(shape)
    distance[1:-1, 1:-1, 1:-1] = 1
    distance = ndimage.morphology.distance_transform_edt(distance)

    zz, yy, xx = np.indices(shape)

    for i in range(ball):
        print("ball{}".format(i))
        if distance.max() < r_min:
            break
        r = np.random.rand()*np.min((distance.max()/2, r_max))
        if r < r_min:
            r = r_min
        index = np.random.randint((distance > r).sum())
        z = zz[distance > r].reshape(-1)[index]
        y = yy[distance > r].reshape(-1)[index]
        x = xx[distance > r].reshape(-1)[index]

        output[(xx-x)**2 + (yy-y)**2 + (zz-z)**2 <= r**2] = 1
        label[output != 0] = 1
        label[z, y, x] = -1

        distance = np.zeros(shape)
        distance[1:-1, 1:-1, 1:-1] = 1
        distance[output > 0] = 0
        distance = ndimage.morphology.distance_transform_edt(distance)

    for i in range(cuboid):
        print("cuboid{}".format(i))
        if distance.max() < r_min:
            break
        r = np.random.rand()*distance.max()
        if r < r_min:
            r = r_min
        index = np.random.randint((distance > r).sum())
        cz = zz[distance > r].reshape(-1)[index]
        cy = yy[distance > r].reshape(-1)[index]
        cx = xx[distance > r].reshape(-1)[index]

        dd = np.max(pow(xx[label < 0] - cx, 2) + pow(yy[label < 0] - cy, 2) + pow(zz[label < 0] - cz, 2))*0.8*0.8

        wx = np.random.rand()*(np.min((pow(dd-2*r_min*r_min, 0.5), r_max))-r_min) + r_min
        wy = np.random.rand()*(np.min((pow(dd-wx*wx-r_min*r_min, 0.5), r_max))-r_min) + r_min
        wz = np.random.rand()*(np.min((pow(dd-wx*wx-wy*wy, 0.5), r_max))-r_min) + r_min

        UL = np.zeros((5, 3))
        LR = np.zeros((5, 3))

        UL[0] = np.array((cz-wz, cy-wy, cx-wx))
        LR[0] = np.array((cz + wz, cy + wy, cx + wx))

        a = np.random.rand()*np.pi
        b = np.random.rand()*np.pi
        c = np.random.rand()*np.pi

        UL[1] = rotM_c((a, b, c), (UL[0, 0] - cx, UL[0, 1] - cy, UL[0, 2]-cz)) + np.array((cx, cy, cz))
        UL[2] = rotM_c((a, b, c), (LR[0, 0] - cx, UL[0, 1] - cy, UL[0, 2]-cz)) + np.array((cx, cy, cz))
        UL[3] = rotM_c((a, b, c), (UL[0, 0] - cx, LR[0, 1] - cy, UL[0, 2]-cz)) + np.array((cx, cy, cz))
        UL[4] = rotM_c((a, b, c), (UL[0, 0] - cx, UL[0, 1] - cy, LR[0, 2]-cz)) + np.array((cx, cy, cz))

        LR[1] = rotM_c((a, b, c), (LR[0, 0] - cx, LR[0, 1] - cy, LR[0, 2]-cz)) + np.array((cx, cy, cz))
        LR[2] = rotM_c((a, b, c), (UL[0, 0] - cx, LR[0, 1] - cy, LR[0, 2]-cz)) + np.array((cx, cy, cz))
        LR[3] = rotM_c((a, b, c), (LR[0, 0] - cx, UL[0, 1] - cy, LR[0, 2]-cz)) + np.array((cx, cy, cz))
        LR[4] = rotM_c((a, b, c), (LR[0, 0] - cx, LR[0, 1] - cy, UL[0, 2]-cz)) + np.array((cx, cy, cz))

        image = np.ones(np.prod(shape), dtype=dtype)
        z, y, x = np.indices(shape)
        xyz = np.concatenate((x.reshape((x.size, 1)), y.reshape((y.size, 1)), z.reshape((z.size, 1))), axis=1)
        image[np.array(list(map(functools.partial(np.dot, b=np.cross(UL[2]-UL[1], UL[3]-UL[1])), xyz-UL[1]))) < 0] = 0
        image[np.array(list(map(functools.partial(np.dot, b=np.cross(UL[3]-UL[1], UL[4]-UL[1])), xyz-UL[1]))) < 0] = 0
        image[np.array(list(map(functools.partial(np.dot, b=np.cross(UL[4]-UL[1], UL[2]-UL[1])), xyz-UL[1]))) < 0] = 0
        image[np.array(list(map(functools.partial(np.dot, b=np.cross(LR[2]-LR[1], LR[3]-LR[1])), xyz-LR[1]))) > 0] = 0
        image[np.array(list(map(functools.partial(np.dot, b=np.cross(LR[3]-LR[1], LR[4]-LR[1])), xyz-LR[1]))) > 0] = 0
        image[np.array(list(map(functools.partial(np.dot, b=np.cross(LR[4]-LR[1], LR[2]-LR[1])), xyz-LR[1]))) > 0] = 0
        image = image.reshape(shape)

        output[image > 0] = 1
        label[(image > 0) & (abs(label) != 1)] = 2
        label[cz, cy, cx] = -2

    label[label < 0] *= -1

    return output, label


class MakeDataset(dataset.DatasetMixin):

    def __init__(self, image, label, shape, dtype=np.float32, label_dtype=np.int32):
        self._image = image  # shape:(imageno, NSlice, height, width)
        self._label = label  # shape:(imageno, NSlice, height, width)
        self._shape = shape  # patchのshape
        self._dtype = dtype
        self._label_dtype = label_dtype

    def __len__(self):
        z, y, x = self._shape//2
        return int((self._label[:, z:self._label.shape[1] - self._shape[0] + z, y:self._label.shape[2] - self._shape[1] + y, x:self._label.shape[3] - self._shape[2] + x] > 0).sum())

    def get_example(self, i):

        no, zz, yy, xx = np.indices(self._image.shape)
        z, y, x = self._shape//2
        no = no[:, z:self._label.shape[1] - self._shape[0] + z, y:self._label.shape[2] - self._shape[1] + y, x:self._label.shape[3] - self._shape[2] + x]
        zz = zz[:, z:self._label.shape[1] - self._shape[0] + z, y:self._label.shape[2] - self._shape[1] + y, x:self._label.shape[3] - self._shape[2] + x]
        yy = yy[:, z:self._label.shape[1] - self._shape[0] + z, y:self._label.shape[2] - self._shape[1] + y, x:self._label.shape[3] - self._shape[2] + x]
        xx = xx[:, z:self._label.shape[1] - self._shape[0] + z, y:self._label.shape[2] - self._shape[1] + y, x:self._label.shape[3] - self._shape[2] + x]
        label = self._label[:, z:self._label.shape[1] - self._shape[0] + z, y:self._label.shape[2] - self._shape[1] + y, x:self._label.shape[3] - self._shape[2] + x]

        no = no[label > 0].reshape(-1)[i]
        zz = zz[label > 0].reshape(-1)[i]
        yy = yy[label > 0].reshape(-1)[i]
        xx = xx[label > 0].reshape(-1)[i]

        label = np.array(self._label[no, zz, yy, xx] - 1, dtype=self._label_dtype)

        zz -= z
        yy -= y
        xx -= x

        image = self._image[no:no + 1, zz:zz + self._shape[0], yy:yy + self._shape[1], xx:xx + self._shape[2]].astype(self._dtype)
        return image, label


class CNNND(Chain):
    def __init__(self, ndim, n_units, n_out, k_size, stride1P=2, pad1P=1, k_sizeP=3):
        initializer = initializers.HeNormal()
        initializers.Uniform()
        super(CNNND, self).__init__()
        with self.init_scope():
            self.conv1 = L.ConvolutionND(ndim, n_units[0], n_units[1], k_size, pad=(k_size - 1)//2, initialW=initializer)
            self.conv2 = L.ConvolutionND(ndim, n_units[1], n_units[min(len(n_units) - 1, 2)], k_size, pad=(k_size - 1)//2, initialW=initializer)
            self.conv3 = L.ConvolutionND(ndim, n_units[min(len(n_units) - 1, 2)], n_units[min(len(n_units) - 1, 3)], k_size, pad=(k_size - 1)//2, initialW=initializer)

            self.bn1 = L.BatchNormalization(n_units[1])
            self.bn2 = L.BatchNormalization(n_units[min(len(n_units) - 1, 2)])
            self.bn3 = L.BatchNormalization(n_units[min(len(n_units) - 1, 3)])
            self.bn4 = L.BatchNormalization(n_units[-1])

            self.l1 = L.Linear(None, n_units[-1])  # n_units  - > n_units
            self.l2 = L.Linear(None, n_out)    # n_units  - > n_out

        self._stride = stride1P
        self._pad = pad1P
        self._k_size = k_sizeP

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.max_pooling_nd(h, self._k_size, stride=self._stride, pad=(self._k_size - 1)//2)
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.max_pooling_nd(h, self._k_size, stride=self._stride, pad=(self._k_size - 1)//2)
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.max_pooling_nd(h, self._k_size, stride=self._stride, pad=(self._k_size - 1)//2)
        h = F.relu(self.bn4(self.l1(h)))
        y = self.l2(h)

        return y


def trainCNN(train, train_label, test, test_label,
             kn1=15, kn2=15, kn3=15, kn4=50,
             kernel=3,
             labelN=2,

             trainno=1, testno=1,
             width=100, height=100, NSlice=100,
             ball=5, cuboid=5,

             batchsize4train=100,
             batchsize4val=100,

             patchsize=15,
             alpha=0.00001,
             gpu=0,
             epoch=20,
             sname=""):
    """
    kn1 - 4:filterの数
    labelN：Nclass分類
    batchsize4train/batchsize4val:学習/バリデーションのbatchsize
    patchsize:パッチサイズ shape = (patchsize, patchsize, patchsize)
    alpha:学習効率 0.1 - 0.000001
    gpu:GPU ID (negative value indicates CPU)
    epoch:エポック数
    """
# 必要があれば外部入力に変更
    dimension = 3
    channel = 1
# set model
    n_unit = np.array((channel, kn1, kn2, kn3, kn4)).astype(np.int32)

    model = L.Classifier(CNNND(dimension, n_unit, labelN, kernel))  # 次元, chanel数の変位, 何クラス分類か, kernel

    if gpu >= 0:
        cuda.get_device(gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    optimizer = optimizers.Adam(alpha=alpha)
    optimizer.setup(model)

    train = MakeDataset(train, train_label, np.array((patchsize, patchsize, patchsize)))
    test = MakeDataset(test, test_label, np.array((patchsize, patchsize, patchsize)))

# 並列化可
    train_iter = iterators.SerialIterator(train, batch_size=batchsize4train, repeat=True)
    test_iter = iterators.SerialIterator(test, batch_size=batchsize4val, repeat=False, shuffle=False)
#    train_iter = iterators.MultiprocessIterator(train, batch_size = batchsize4train, repeat = True, n_processes = 2)
#    test_iter = iterators.MultiprocessIterator(test, batch_size = batchsize4val, repeat = False, shuffle = False, n_processes = 2)

    updater = training.StandardUpdater(train_iter, optimizer, device=gpu)
    trainer = training.Trainer(updater, (epoch, 'epoch'))

    trainer.extend(training.extensions.Evaluator(test_iter, model, device=gpu))   # テストしつつ学習するなら必要
    trainer.extend(training.extensions.LogReport(log_name='log' + sname), trigger=(1, 'epoch'))   # logを出力したければ必要
    trainer.extend(training.extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy', 'main/loss', 'validation/main/loss', 'elapsed_time']))   # epochごとの経過をprint
    trainer.extend(training.extensions.ProgressBar(training_length=(epoch, 'epoch')))   # 予測残り時間表示
    trainer.extend(training.extensions.snapshot(filename='snapshot_iter_' + sname + '_{.updater.epoch}'), trigger=(1, 'epoch'))   # epochごとにtrainerを保存したければ必要(途中のepochから再開したい時)
    trainer.extend(training.extensions.snapshot_object(model, 'mdl_snapshot_' + sname + '_{.updater.epoch}'), trigger=(1, 'epoch'))   # epochごとにモデルを保存したければ必要

    trainer.run()


def result(test, label, modelpath, kn1=31, kn2=31, kn3=31, kn4=100, patchsize=31, labelN=2, kernel=3, gpu=0, batchsize=200, output="result"):
    channel = 1
    dimension = 3

    n_unit = np.array((channel, kn1, kn2, kn3, kn4)).astype(np.int32)

    model = L.Classifier(CNNND(dimension, n_unit, labelN, kernel))  # 次元, chanel数の変位, 何クラス分類か, kernel

    if gpu >= 0:
        cuda.get_device(gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    serializers.load_npz(modelpath, model)

    test = MakeDataset(test, label, np.array((patchsize, patchsize, patchsize)))
    test_iter = iterators.SerialIterator(test, batch_size=batchsize, repeat=False, shuffle=False)

    start = 0
    for minibatch in test_iter:
        end = start + len(minibatch)
        in_arrays = dataset.convert.concat_examples(minibatch, gpu)

        prediction = model.predictor(in_arrays[0])
        if start == 0:
            certainties0 = cp.exp(prediction.data[:, 0])/(cp.exp(prediction.data[:, 1]) + cp.exp(prediction.data[:, 0]))
            certainties1 = cp.exp(prediction.data[:, 1])/(cp.exp(prediction.data[:, 1]) + cp.exp(prediction.data[:, 0]))
        else:
            certainties0 = cp.concatenate((certainties0, cp.exp(prediction.data[:, 0])/(cp.exp(prediction.data[:, 1]) + cp.exp(prediction.data[:, 0]))))
            certainties1 = cp.concatenate((certainties1, cp.exp(prediction.data[:, 1])/(cp.exp(prediction.data[:, 1]) + cp.exp(prediction.data[:, 0]))))
        start = end
        print("{:.03f}%".format(100.0 * float(start) / np.sum(label != -1)), end="")

    outimage0 = np.zeros_like(label)
    outimage1 = np.zeros_like(label)
    z = y = x = patchsize//2

    mask = label[:, z:label.shape[1] - patchsize + z, y:label.shape[2] - patchsize + y, x:label.shape[3] - patchsize + x] != 0
    outimage0[:, z:label.shape[1] - patchsize + z, y:label.shape[2] - patchsize + y, x:label.shape[3] - patchsize + x][mask] = cuda.to_cpu(certainties0[:np.sum(mask)])
    outimage1[:, z:label.shape[1] - patchsize + z, y:label.shape[2] - patchsize + y, x:label.shape[3] - patchsize + x][mask] = cuda.to_cpu(certainties1[:np.sum(mask)])

    outimage0.tofile("{}_0.raw".format(output))
    outimage1.tofile("{}_1.raw".format(output))


def main():
    parse = argparse.ArgumentParser(description='Chainer example: 〇◆classification')
    parse.add_argument('--epoch', '-e', type=int, default=10,
                       help='Number of sweeps over the dataset to train')
    parse.add_argument('--batchsize4train', type=int, default=100,
                       help='batch size for training')
    parse.add_argument('--batchsize4val', type=int, default=100,
                       help='batch size for validation')
    parse.add_argument('--trainno', type=int, default=10,
                       help='case no for training')
    parse.add_argument('--testno', type=int, default=5,
                       help='case no for testing')
    parse.add_argument('--NSlice', type=int, default=64,
                       help="image's Slice No.")
    parse.add_argument('--height', type=int, default=64,
                       help='image height')
    parse.add_argument('--width', type=int, default=64,
                       help='image width')
    parse.add_argument('--ball', type=int, default=5,
                       help='ball Count')
    parse.add_argument('--cuboid', type=int, default=5,
                       help='cuboid Count.')

    parse.add_argument('--patchsize', '-p', type=int, default=15,
                       help='patch size')
    parse.add_argument('--kn1', type=int, default=15,
                       help='filter no1')
    parse.add_argument('--kn2', type=int, default=15,
                       help='filter no2')
    parse.add_argument('--kn3', type=int, default=15,
                       help='filter no3')
    parse.add_argument('--fcn', type=int, default=50,
                       help='fully connection no')
    parse.add_argument('--alpha', type=float, default=0.00001,
                       help='alpha')

    parse.add_argument('--gpu', '-g', type=int, default=0,
                       help='GPU ID (negative value indicates CPU)')
    args = parse.parse_args()

    sname = "kn_{kn1}_{kn2}_{kn3}_{kn4}_PS{patchsize}".format(kn1=args.kn1, kn2=args.kn2, kn3=args.kn3, kn4=args.fcn, patchsize=args.patchsize)

    print("make dataset")
    image = np.zeros((args.trainno + args.testno, args.NSlice, args.height, args.width))
    label = np.zeros((args.trainno + args.testno, args.NSlice, args.height, args.width))
    shape = np.array((args.NSlice, args.height, args.width))

    for i in range(args.trainno + args.testno):
        image[i], label[i] = MakeImage(shape, args.ball, args.cuboid)

    label[:args.trainno].tofile("train_label.raw")
    image[:args.trainno].tofile("train.raw")
    label[args.trainno:].tofile("test_label.raw")
    image[args.trainno:].tofile("test.raw")

    print("CNN start")
    trainCNN(image[:args.trainno], label[:args.trainno], image[args.trainno:], label[args.trainno:],
             kn1=args.kn1, kn2=args.kn2, kn3=args.kn3, kn4=args.fcn,
             batchsize4train=args.batchsize4train, batchsize4val=args.batchsize4val,
             patchsize=args.patchsize, alpha=args.alpha, gpu=args.gpu, epoch=args.epoch, sname=sname)

    print("CNN result")
    result(image[args.trainno:], label[args.trainno:],
           modelpath="result/mdl_snapshot_{sname}_{epoch}".format(sname=sname, epoch=args.epoch),
           kn1=args.kn1, kn2=args.kn2, kn3=args.kn3, kn4=args.fcn,
           patchsize=args.patchsize, gpu=args.gpu, batchsize=args.batchsize4val)


if __name__ == '__main__':
    main()
