import numpy as np
from skimage.util.shape import view_as_windows


class nn_convolutional_layer:

    def __init__(self, Wx_size, Wy_size, input_size, in_ch_size, out_ch_size, std=1e0):

        # Xavier init
        self.W = np.random.normal(0, std / np.sqrt(in_ch_size * Wx_size * Wy_size / 2),
                                  (out_ch_size, in_ch_size, Wx_size, Wy_size))
        self.b = 0.01 + np.zeros((1, out_ch_size, 1, 1))
        self.input_size = input_size

    def update_weights(self, dW, db):
        self.W += dW
        self.b += db

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

    def forward(self, x):
        y = view_as_windows(x, (x.shape[0], x.shape[1], self.W.shape[2], self.W.shape[3]))
        # print('y shape : ', y.shape)

        # y.reshape (1, 1, w_size, h_size, batch_size, -1) // -1 mean in_ch_size * filter_width * filter_height
        y = y.reshape(y.shape[0], y.shape[1], y.shape[2], y.shape[3], y.shape[4], -1)

        # filter reshape w.reshape(num_filter, -1) // -1 mean in_ch_size * filter_width * filter_height
        ft = self.W.reshape(self.W.shape[0], -1)

        # return out as (1,1,w_size, h_size, batch_size, num_filter)
        out = y @ ft.T
        # print ('out shape : ', out.shape)

        # return out (1,1,w_size, h_size, batch_size, num_filter) -> (w_size, h_size, batch_size, num_filter)
        out = out.squeeze(axis=(0,1))
        # print('out shape after squeeze: ', out.shape)

        # for plus b -> b is just depend num_filter // 생각하기 쉽게 순서를 좀 바꿔주자
        # batch_size, num_filter_size, w_size, h_size
        out = np.transpose(out, (2, 3, 0, 1))
        # print('out shape after transpose: ', out.shape)

        # apply bias
        out = out + self.b
        # print('out shape : ', out.shape)


        return out

    def backprop(self, x, dLdy):
        # calc dLdx
        # dLdy padding
        # print ('input dLdy : ', dLdy.shape, '\n')
        # set padding size, only axis out_width and out_height
        pad_size = int(np.floor((self.W.shape[2] + 1) / 2))
        # pad only axis out_width and out_height with pad_size and value 0
        #       axis=0, axis=1, axis=2              , axis=3
        npad = ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size))
        dLdy_pad = np.pad(dLdy, npad, 'constant', constant_values=(0))
        # print ('dLdy_pad shape : ', dLdy_pad.shape)

        # view dLdy_pad as window (batch_size, num_filter, filter_width, filter_height)
        dLdy_view = view_as_windows(dLdy_pad,
                                    (dLdy_pad.shape[0], dLdy_pad.shape[1], self.W.shape[2], self.W.shape[3]))
        # print('dLdy_view shape : ', dLdy_view.shape)

        # W reverse order > keep batch_size, in_ch_size
        w_rev = np.flip(self.W, (2, 3))

        # flat dLdy_view and w_rev
        dLdy_view = dLdy_view.reshape(dLdy_view.shape[0], dLdy_view.shape[1], dLdy_view.shape[2], dLdy_view.shape[3],
                                      dLdy_view.shape[4], -1)
        # print('dLdy_view shape after reshape : ', dLdy_view.shape)
        # tranpose로 순서 변경 in_ch_size, num_filter, filter_width, filter_heigh
        w_rev = w_rev.transpose(1, 0, 2, 3)
        # in_ch_size 외 계산이 쉽도록 flatten
        w_rev = w_rev.reshape(w_rev.shape[0], -1)
        # print ('w_rev shape after reshape: ', w_rev.shape)

        # convolution dLdy * w_reverse.T
        dLdx = dLdy_view @ w_rev.T
        # print('dLdx shape : ', dLdx.shape)
        dLdx = dLdx.squeeze(axis=(0,1))
        # dLdx = dLdx.reshape(dLdx.shape[0], dLdx.shape[1], dLdx.shape[2], 1)
        # print ('dLdx shape : ', dLdx.shape)
        dLdx = dLdx.transpose(2, 3, 0, 1)

        # calc dLdw
        # x.shape = (batch size, input channel size, in width, in height) (8,3,32,32)
        # dLdy.shape = (batch size, num filter, out width, out height) (8,8,30,30)
        # dLdW.shape = (num_filters, in_ch_size, filter_width, filter_height)
        # convolution x * dLdy
        x_view = view_as_windows(x, (dLdy.shape[0], self.W.shape[1], dLdy.shape[2], dLdy.shape[3]))
        # print ('x_view shape : ', x_view.shape)
        # in_ch_size, view_out_width, view_out_height, batch_size, filter(dLdy)_width, filter(dLdy)_height
        x_view = x_view.transpose(0, 1, 5, 2, 3, 4, 6, 7)
        # print ('x_view shape after transpose: ', x_view.shape)
        # flat x_view keep in_ch_size, view_out_width, view_out_height
        x_view = x_view.reshape(x_view.shape[0], x_view.shape[1], x_view.shape[2], x_view.shape[3], x_view.shape[3], -1)
        # print('x_view shape after reshape: ', x_view.shape)
        # flat dLdy keep num filters
        dLdy_flat = dLdy.transpose(1, 0, 2, 3)
        dLdy_flat = dLdy_flat.reshape(dLdy_flat.shape[0], -1)
        # print ('dLdy_flat shape : ', dLdy_flat.shape)
        dLdW = x_view @ dLdy_flat.T
        # print('dLdW shape :', dLdW.shape)
        dLdW = dLdW.squeeze(axis=(0,1))
        # dLdW = dLdW.reshape(dLdW.shape[0], dLdW.shape[1], dLdW.shape[2], 1)
        dLdW = dLdW.transpose(3, 0, 1, 2)
        # print('dLdW shape :', dLdW.shape)

        # calc dLdb
        # sum dLdy keep num_filter = axis1
        dLdb = np.sum(dLdy, (0, 2, 3))
        dLdb = dLdb.reshape(self.b.shape)
        # print ('dLdb shape : ', dLdb.shape)

        return dLdx, dLdW, dLdb


class nn_max_pooling_layer:
    def __init__(self, stride, pool_size):
        self.stride = stride
        self.pool_size = pool_size

    def forward(self, x):
        y = view_as_windows(x, (x.shape[0], x.shape[1], self.pool_size, self.pool_size), self.stride)
        # y.shape = (1, 1, 16, 16, 8, 3, 2, 2)
        # print ('[mp F]y shape: ', y.shape)
        y = y.squeeze(axis=(0,1))
        # [mp F]y shape after squeeze:  (16, 16, 8, 3, 2, 2)
        # print('[mp F]y shape after squeeze: ', y.shape)
        y = y.reshape(y.shape[0], y.shape[1], y.shape[2], y.shape[3], -1)
        # [mp F]y shape after reshape:  (16, 16, 8, 3, 4)
        # print('[mp F]y shape after reshape: ', y.shape)
        out = y.max(axis=4)
        # [mp F]out shape after max:  (16, 16, 8, 3)
        # print('[mp F]out shape after max: ', out.shape)
        self.max_index = y.argmax(axis=4)
        self.max_index = self.max_index.transpose(2, 3, 0, 1)
        # [mp F]max_index shape : (8, 3, 16, 16)
        # print('[mp F]max_index shape :', self.max_index.shape)
        # print('[mp F]max_index :', self.max_index[0][0])

        out = out.transpose(2, 3, 0, 1)
        return out

    def backprop(self, x, dLdy):
        tdx = np.zeros_like(x).astype(float)
        for i in np.arange(self.max_index.shape[0]):
            for j in np.arange(self.max_index.shape[1]):
                for k in np.arange(self.max_index.shape[2]):
                    for l in np.arange(self.max_index.shape[3]):
                        n = int(np.floor(self.max_index[i, j, k, l] / self.pool_size))
                        m = int(self.max_index[i, j, k, l] % self.pool_size)
                        # print ('n,m = (',n,',', m,')  ', self.max_index[i,j,k,l])
                        tdx[i, j, self.pool_size * k + n, self.pool_size * l + m] = dLdy[i, j, k, l]

        dLdx = tdx

        return dLdx

class nn_activation_layer:

    # linear layer. creates matrix W and bias b
    # W is in by out, and b is out by 1
    def __init__(self):
        pass

    def forward(self, x):
        return np.maximum(x, 0)

    def backprop(self, x, dLdy):
        return (x >= 0).astype('uint8') * dLdy


# fully connected layer
class nn_fc_layer:

    def __init__(self, input_size, output_size, filt_size=1, std=1):
        # Xavier-He init
        self.W = np.random.normal(0, std/np.sqrt(input_size*filt_size**2/2), (output_size, input_size, filt_size, filt_size))
        self.b=0.01+np.zeros((1,output_size,1,1))

    def forward(self,x):
        # compute forward pass of given parameter
        # print('[nn_fc_layer] x.shape : ', x.shape, 'W.shape :', self.W.shape)
        output_size = self.W.shape[0]
        batch_size = x.shape[0]
        Wx = np.dot(x.reshape((batch_size, -1)),(self.W.reshape(output_size, -1)).T)
        return np.expand_dims(np.expand_dims(Wx,axis=2),axis=3)+self.b

    def backprop(self,x,dLdy):

        dLy = np.expand_dims(dLdy,axis=4)
        dyx = np.expand_dims(self.W,axis=0)

        dLdx = np.sum(dLy*dyx,axis=1).reshape(x.shape)

        dLdW=np.expand_dims(x,axis=1)*dLy
        dLdW=np.sum(dLdW,axis=0)

        dLdb=dLdy*np.ones(dLdy.shape)
        dLdb=(np.sum(dLdb,axis=0,keepdims=True))

        return dLdx,dLdW,dLdb

    def update_weights(self,dLdW,dLdb):

        # parameter update
        self.W=self.W+dLdW
        self.b=self.b+dLdb

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b


class nn_softmax_layer:

    def __init__(self):
        pass

    def forward(self, x):
        s = x - np.amax(x, axis=1)[:, np.newaxis]
        return (np.exp(s) / np.sum(np.exp(s), axis=1)[:, np.newaxis]).reshape((x.shape[0],x.shape[1]))

    def backprop(self, x, dLdy):
        # getting input dLdy of dimension n x len(x)
        # should return dLdx

        p = self.forward(x)
        dLdx = -np.sum(p * dLdy, axis=1, keepdims=True) * p + p * dLdy

        return dLdx.reshape(x.shape)


class nn_cross_entropy_layer:

    def __init__(self):
        self.eps=1e-15

    def forward(self, x, y):

        batch_size = y.shape[0]
        num_class = x.shape[1]
        onehot = np.zeros((batch_size, num_class))
        onehot[range(batch_size), (np.array(y)).reshape(-1, )] = 1

        # to avoid numerial instability
        x[x<self.eps]=self.eps
        x=x/np.sum(x,axis=1)[:,np.newaxis]

        return sum(-np.sum(np.log(np.array(x).reshape(batch_size, -1)) * onehot, axis=0)) / batch_size

    def backprop(self, x, y):
        # print('nn_ce x :', x.shape, x)
        batch_size = x.shape[0]
        p = np.zeros(x.shape)
        p[range(batch_size), y.reshape((batch_size,))] = -1 / x[range(batch_size), y.reshape((batch_size,))]

        return p / batch_size
