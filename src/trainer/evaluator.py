from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, r2_score
import pandas as pd
import os
import numpy as np

import matplotlib.pyplot as plt

import time
import piexif

import json

from sklearn.metrics import roc_curve, auc,f1_score


import sys
sys.path.append('../../scripts/')
from settings import settings

try:
    import torch
except ImportError:
    torch = None

### Evaluator for graph classification
class Evaluator:
    def __init__(self, name, eval_metric, n_tasks, mean=None, std=None):
        self.name = name
        self.eval_metric = eval_metric
        self.n_tasks = n_tasks
        self.mean = mean
        self.std = std

    def _parse_and_check_input(self, y_true, y_pred, valid_ids=None):
        # converting to torch.Tensor to numpy on cpu
        if torch is not None and isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if torch is not None and isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        ## check type
        if not isinstance(y_true, np.ndarray):
            raise RuntimeError('Arguments to Evaluator need to be either numpy ndarray or torch tensor')

        if not y_true.shape == y_pred.shape:
            raise RuntimeError('Shape of y_true and y_pred must be the same')

        if not y_true.ndim == 2:
            raise RuntimeError('y_true and y_pred mush to 2-dim arrray, {}-dim array given'.format(y_true.ndim))

        if not y_true.shape[1] == self.n_tasks:
            raise RuntimeError('Number of tasks for {} should be {} but {} given'.format(self.name, self.n_tasks, y_true.shape[1]))
        if valid_ids is not None and isinstance(valid_ids, torch.Tensor):
            valid_ids = valid_ids.detach().cpu().numpy()
        return y_true, y_pred, valid_ids


    def eval(self, y_true, y_pred, valid_ids=None,training_set_name=None):
        y_true, y_pred, valid_ids = self._parse_and_check_input(y_true, y_pred, valid_ids)
        if self.eval_metric == 'rocauc':
            return self._eval_rocauc(y_true, y_pred,training_set_name)
        elif self.eval_metric == 'rocauc_resp':
            return self._eval_rocauc_resp(y_true, y_pred, valid_ids)
        elif self.eval_metric == 'ap':
            return self._eval_ap(y_true, y_pred)
        elif self.eval_metric == 'ap_resp':
            return self._eval_ap_resp(y_true, y_pred)
        elif self.eval_metric == 'rmse':
            return self._eval_rmse(y_true, y_pred,training_set_name)
        elif self.eval_metric == 'acc':
            return self._eval_acc(y_true, y_pred)
        elif self.eval_metric == 'mae':
            return self._eval_mae(y_true, y_pred)
        elif self.eval_metric == 'r2':
            return self._eval_r2(y_true, y_pred)
        else:
            raise ValueError('Undefined eval metric %s ' % (self.eval_metric))

    def _eval_rocauc(self, y_true, y_pred,training_set_name=None):      
                    
        '''
            compute ROC-AUC averaged across tasks
        '''

        rocauc_list = []

        for i in range(y_true.shape[1]):
            #AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
                # ignore nan values
                is_labeled = y_true[:,i] == y_true[:,i]
            #    print(y_pred[is_labeled,i])
                # compute AUC with roc_curve
                fpr, tpr, _ = roc_curve(y_true[is_labeled,i], y_pred[is_labeled,i])
                rocauc_list.append(auc(fpr, tpr))

                # plot distribution of y_pred with color given by y_true
                fig,ax=plt.subplots(figsize=(10,10))

                plt.scatter(np.arange(len(y_pred[is_labeled,i])), y_pred[is_labeled,i], c=y_true[is_labeled,i], label=['0','1']
                            
                             )

                ax.tick_params(axis="y",direction="in", pad=-22)
                ax.tick_params(axis="x",direction="in", pad=-15)
               #  fig.tight_layout()
                
                task=settings[self.name]['output_names'][i]

                #get x_scale and y_scale from automaticaly fitted plot
                x_scale=fig.axes[0].get_xlim()
                y_scale=fig.axes[0].get_ylim()
                
                #add axis labels
                # plt.xlabel('sample')
                # plt.ylabel('prediction')

                plt.margins(x=0,y=0)


                img_path=f'/home/zach/whc_backend/web/greens/static/distribs/{self.name}_{task}_{training_set_name}.jpg'
                fig.savefig(img_path)
                plt.close(fig)
                metadata=piexif.load(img_path)

                metadata['Exif'][piexif.ExifIFD.UserComment]=json.dumps(
                    {
                        "x_scale":x_scale,
                        "y_scale":y_scale,
                        "type":"classification",
                        "roc_auc":float(rocauc_list[-1]),
                #        "unit":None
                    }
                ).encode()

                
                
              # 
                piexif.insert(piexif.dump(metadata), img_path)




                
                


                

                #get boundary for best f1 score
                # thresholds= np.linspace( np.min(y_pred[is_labeled,i]), np.max(y_pred[is_labeled,i]), 1000)
                # f1_list = []
                # for threshold in thresholds:
                #     y_pred_tmp = np.zeros_like(y_pred[:,i])
                #     y_pred_tmp[y_pred[:,i] > threshold] = 1
                #     f1_list.append(f1_score(y_true[:,i], y_pred_tmp))
                
                

                

                # best_f1 = np.argmax(f1_list)
                # print('best f1 threshold: ', thresholds[best_f1], 'f1 score', f1_list[best_f1])

                



        if len(rocauc_list) == 0:
            raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

        return sum(rocauc_list)/len(rocauc_list)

    def _eval_rocauc_resp(self, y_true, y_pred, valid_ids=None):
        '''
            compute ROC-AUC averaged across tasks
        '''

        rocauc_list = []

        for i in range(y_true.shape[1]):
            #AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
                # ignore nan values
                is_labeled = y_true[:,i] == y_true[:,i]
                if valid_ids is not None:
                    is_labeled = np.logical_and(is_labeled, valid_ids[:, i])
                if len(y_true[is_labeled,i] != 0):
                    rocauc_list.append(roc_auc_score(y_true[is_labeled,i], y_pred[is_labeled,i]))

        if len(rocauc_list) == 0:
            raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

        return rocauc_list


    def _eval_ap(self, y_true, y_pred):
        '''
            compute Average Precision (AP) averaged across tasks
        '''

        ap_list = []

        for i in range(y_true.shape[1]):
            #AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
                # ignore nan values
                is_labeled = y_true[:,i] == y_true[:,i]
                if len(y_true[is_labeled,i] != 0):
                    ap = average_precision_score(y_true[is_labeled,i], y_pred[is_labeled,i])

                    ap_list.append(ap)

        if len(ap_list) == 0:
            raise RuntimeError('No positively labeled data available. Cannot compute Average Precision.')

        return sum(ap_list)/len(ap_list)
    def _eval_ap_resp(self, y_true, y_pred):
        '''
            compute Average Precision (AP) averaged across tasks
        '''

        ap_list = []

        for i in range(y_true.shape[1]):
            #AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
                # ignore nan values
                is_labeled = y_true[:,i] == y_true[:,i]
                ap = average_precision_score(y_true[is_labeled,i], y_pred[is_labeled,i])

                ap_list.append(ap)

        if len(ap_list) == 0:
            raise RuntimeError('No positively labeled data available. Cannot compute Average Precision.')

        return ap_list

    def _eval_rmse(self, y_true, y_pred,training_set_name=None):
        '''
            compute RMSE score averaged across tasks
        '''
        rmse_list = []
        for i in range(y_true.shape[1]):
            # ignore nan values
            is_labeled = y_true[:,i] == y_true[:,i]
            if (self.mean is not None) and (self.std is not None):
                rmse_list.append(np.sqrt(((y_true[is_labeled,i] - (y_pred[is_labeled,i]*self.std[i]+self.mean[i]))**2).mean()))
            else:
                rmse_list.append(np.sqrt(((y_true[is_labeled,i] - y_pred[is_labeled,i])**2).mean()))
            # plot distribution (y_true vs y_pred)
            plt.figure()

            fig,ax =plt.subplots(
                figsize=(10,10),
            )

            plt.scatter(y_true[is_labeled,i], y_pred[is_labeled,i], alpha=0.5,label=['true','pred'])

            #  fig.tight_layout()
            plt.margins(x=0,y=0)
            task=settings[self.name]['output_names'][i]


            x_scale=fig.axes[0].get_xlim()
            y_scale=fig.axes[0].get_ylim()

            #get x_scale and y_scale from automaticaly fitted plot

            ax.tick_params(axis="y",direction="in", pad=-22)
            ax.tick_params(axis="x",direction="in", pad=-15)
        
            # #add axis labels
            # plt.xlabel(f'{task} true')
            # plt.ylabel(f'{task} pred')

            # plt.title(f'{task} true vs pred')


            img_path=f'/home/zach/whc_backend/web/greens/static/distribs/{self.name}_{task}_{training_set_name}.jpg'
            fig.savefig(img_path)
            plt.close(fig)
            metadata=piexif.load(img_path)

            metadata['Exif'][piexif.ExifIFD.UserComment]=json.dumps(
                {
                    "x_scale":x_scale,
                    "y_scale":y_scale,
                    "type":"regression",
                    "rmse":float(rmse_list[-1]),
                    "unit":settings[self.name]['units'],



                }
            ).encode()



            
            # 
            piexif.insert(piexif.dump(metadata), img_path)


        return sum(rmse_list)/len(rmse_list)
    
    def _eval_mae(self, y_true, y_pred):
        '''
            compute MAE score averaged across tasks
        '''
        mae_list = []

        for i in range(y_true.shape[1]):
            # ignore nan values
            is_labeled = y_true[:,i] == y_true[:,i]
            if (self.mean is not None) and (self.std is not None):
                mae_list.append(mean_absolute_error(y_true[:,i], y_pred[:,i]*self.std[i]+self.mean[i]))
            else:
                mae_list.append(mean_absolute_error(y_true[:,i], y_pred[:,i]))

        return sum(mae_list)/len(mae_list)
    def _eval_r2(self, y_true, y_pred):
        '''
            compute R2 score averaged across tasks
        '''
        r2_list = []

        for i in range(y_true.shape[1]):
            # ignore nan values
            is_labeled = y_true[:,i] == y_true[:,i]
            if (self.mean is not None) and (self.std is not None):
                r2_list.append(r2_score(y_true[is_labeled,i], y_pred[is_labeled,i]*self.std[i]+self.mean[i]))
            else:
                r2_list.append(r2_score(y_true[is_labeled,i], y_pred[is_labeled,i]))

        return r2_list

    def _eval_acc(self, y_true, y_pred):
        acc_list = []

        for i in range(y_true.shape[1]):
            is_labeled = y_true[:,i] == y_true[:,i]
            correct = y_true[is_labeled,i] == y_pred[is_labeled,i]
            acc_list.append(float(np.sum(correct))/len(correct))

        return sum(acc_list)/len(acc_list)