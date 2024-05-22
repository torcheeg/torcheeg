import torch.nn as nn
import torch.nn.functional as F
import torch

class ATCNet(nn.Module):
    r'''
    ATCNet: An attention-based temporal convolutional network forEEG-based motor imagery classiﬁcation.For more details ,please refer to the following information:

    - Paper: H. Altaheri, G. Muhammad and M. Alsulaiman, "Physics-Informed Attention Temporal Convolutional Network for EEG-Based Motor Imagery Classification," in IEEE Transactions on Industrial Informatics, vol. 19, no. 2, pp. 2249-2258, Feb. 2023, doi: 10.1109/TII.2022.3197419.
    - URL: https://github.com/Altaheri/EEG-ATCNet

    .. code-block:: python
        
        import torch
        
        from torcheeg.models import ATCNet

        model = ATCNet(in_channels=1,
                       num_classes=4,
                       num_windows=3,
                       num_electrodes=22,
                       chunk_size=128)

        input = torch.rand(2, 1, 22, 128) # (batch_size, in_channels, num_electrodes,chunk_size) 
        output = model(input)

    Args:
        in_channels (int): The number of channels of the signal corresponding to each electrode. If the original signal is used as input, in_channels is set to 1; if the original signal is split into multiple sub-bands, in_channels is set to the number of bands. (default: :obj:`1`)
        num_electrodes (int): The number of electrodes. (default: :obj:`32`)
        num_classes (int): The number of classes to predict. (default: :obj:`4`)
        num_windows (int): The number of sliding windows after conv block. (default: :obj:`3`)
        num_electrodes (int): The number of electrodes if the input is EEG signal. (default: :obj:`22`)
        conv_pool_size (int):  The size of the second average pooling layer kernel in the conv block. (default: :obj:`7`)
        F1 (int): The channel size of the temporal feature maps in conv block. (default: :obj:`16`)
        D (int): The number of second conv layer's filters linked to each temporal feature map in the previous layer in conv block. (default: :obj:`2`)
        tcn_kernel_size (int): The size of conv layers kernel in the TCN block. (default: :obj:`4`)
        tcn_depth (int): The times of TCN loop. (default: :obj:`2`)
        chunk_size (int): The Number of data points included in each EEG chunk. (default: :obj:`1125`)
    '''
    def __init__(self,
                    in_channels: int = 1,
                    num_classes: int = 4,
                    num_windows: int = 3,
                    num_electrodes: int = 22,
                    conv_pool_size: int = 7,
                    F1: int = 16,
                    D: int =2,
                    tcn_kernel_size: int = 4,
                    tcn_depth: int = 2,
                    chunk_size: int = 1125,
                    ):  
        super(ATCNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_windows = num_windows
        self.num_electrodes = num_electrodes
        self.pool_size = conv_pool_size
        self.F1 = F1
        self.D = D
        self.tcn_kernel_size = tcn_kernel_size
        self.tcn_depth = tcn_depth
        self.chunk_size = chunk_size
        F2 = F1*D

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels,F1,(1,int(chunk_size/2+1)),stride=1,padding = 'same',bias=False),
            nn.BatchNorm2d(F1, False),
            nn.Conv2d(F1,F2,(num_electrodes,1),padding = 0,groups=F1),
            nn.BatchNorm2d(F2,False),
            nn.ELU(),
            nn.AvgPool2d((1,8)),
            nn.Dropout2d(0.1),
            nn.Conv2d(F2,F2,(1,16),bias=False,padding='same'),
            nn.BatchNorm2d(F2,False),
            nn.ELU(),
            nn.AvgPool2d((1,self.pool_size)),
            nn.Dropout2d(0.1)
        )
        self.__build_model()
        
    def __build_model(self):
        with torch.no_grad():
            x = torch.zeros(2,self.in_channels,self.num_electrodes,self.chunk_size)
            x = self.conv_block(x)
            x = x[:,:,-1,:]
            x = x.permute(0,2,1)
            self.__chan_dim,self.__embed_dim = x.shape[1:]
            self.win_len = self.__chan_dim - self.num_windows +1

            for i in range(self.num_windows):
                st = i 
                end = x.shape[1]  -self.num_windows+i+1
                x2 = x[:,st:end,:]

                self.__add_msa(i)
                x2_= self.get_submodule("msa"+str(i))(x2,x2,x2)[0]
                self.__add_msa_drop(i) 
                x2_ = self.get_submodule("msa_drop"+str(i))(x2)
                x2 = torch.add(x2,x2_)
                
                for j in range(self.tcn_depth):
                    self.__add_tcn((i+1)*j,x2.shape[1])
                    out = self.get_submodule("tcn"+str( (i+1)*j ))(x2)
                    if x2.shape[1] != out.shape[1]: 
                        self.__add_recov(i)                   
                        x2 = self.get_submodule("re"+str(i))(x2)
                    x2 = torch.add(x2,out)
                    x2 = nn.ELU()(x2) 
                x2 = x2[:,-1,:]
                self.__dense_dim = x2.shape[-1]
                self.__add_dense(i)
                x2 = self.get_submodule("dense"+str(i))(x2)

   
    def __add_msa(self,index:int):
        
        self.add_module('msa'+str(index),nn.MultiheadAttention(
                                         embed_dim=self.__embed_dim,
                                         num_heads=2,
                                         batch_first=True))
    def __add_msa_drop(self,index):
        self.add_module('msa_drop'+str(index),nn.Dropout(0.3))

    def __add_tcn(self,index:int,num_electrodes:int):
        self.add_module('tcn'+str(index), 
           nn.Sequential(
            nn.Conv1d(num_electrodes,32,self.tcn_kernel_size,padding='same'),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Conv1d(32,32,self.tcn_kernel_size,padding = 'same'),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(0.3) )
        )

    def __add_recov(self,index:int):
        self.add_module('re'+str(index),nn.Conv1d(self.win_len,32,4,padding='same'))

    def __add_dense(self, index:int):
        self.add_module('dense'+str(index),nn.Linear(self.__dense_dim,self.num_classes))

    def forward(self,x):
        r'''
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, 22, 1125]`. Here, :obj:`n` corresponds to the batch size, :obj:`22` corresponds to :obj:`num_electrodes`, and :obj:`1125` corresponds to :obj:`chunk_size`.

        Returns:
            torch.Tensor[size of batch, number of classes]: The predicted probability that the samples belong to the classes.
        '''
        x = self.conv_block(x)
        x = x[:,:,-1,:]
        x = x.permute(0,2,1)

        
        for i in range(self.num_windows):
            st = i 
            end = x.shape[1] -self.num_windows+i+1
            x2 = x[:,st:end,:]
            x2_= self.get_submodule("msa"+str(i))(x2,x2,x2)[0] 
            x2_ = self.get_submodule("msa_drop"+str(i))(x2)
            x2 = torch.add(x2,x2_)
            

            for j in range(self.tcn_depth):
               out = self.get_submodule("tcn"+str( (i+1)*j ))(x2)
               if x2.shape[1] != out.shape[1]:                 
                    x2 = self.get_submodule("re"+str(i))(x2)
               x2 = torch.add(x2,out)
               x2 = nn.ELU()(x2) 
            x2 = x2[:,-1,:]
            x2 = self.get_submodule("dense"+str(i))(x2)
            if i == 0:
                sw_concat = x2
            else:
                sw_concat =sw_concat.add(x2)

        x = sw_concat/self.num_windows
        x = nn.Softmax(dim=1)(x)
        return x
