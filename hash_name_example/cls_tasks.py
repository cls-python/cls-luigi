import hashlib
import luigi
from collections.abc import Iterable
from pathlib import Path
import sys
from os.path import dirname
from os import makedirs
from os.path import join as pjoin


sys.path.append('../')
from inhabitation_task import LuigiCombinator
    

class CLSBaseTask():
    global_hash_path = luigi.Parameter(default="results/hash_files/")
    
    @staticmethod
    def _get_flatten_data(data):
        flattened_data = []
        if isinstance(data, luigi.target.FileSystemTarget):
            flattened_data.append(data)

        elif isinstance(data, (list, tuple)):
            for item in data:
                flattened_data.extend(CLSBaseTask._get_flatten_data(item))
        elif isinstance(data, dict):
            for _ , value in data.items():
                flattened_data.extend(CLSBaseTask._get_flatten_data(value))
        else:
            # error case ? just throw away?
            pass

        return flattened_data
    
    def _get_variant_filename(self, name= ""):
        if not name: 
            name = self.__class__.__name__ + "_result"
        makedirs(dirname(self.global_hash_path), exist_ok=True)
        hash_value = hashlib.md5()
        label = ""     
        
        if isinstance(self.input(), luigi.target.FileSystemTarget):
            label =  Path(self.input().path).stem +  " --> " + self.__class__.__name__ if len(Path(self.input().path).stem) > 0 else self.__class__.__name__ 

        elif isinstance(self.input(), (list, tuple, dict)):
            
            flattened_data = CLSBaseTask._get_flatten_data(self.input())
            var_label_name = []
            for item in flattened_data:
                var_label_name.append(Path(item.path).stem)
            label = "(" + (", ".join(var_label_name)) + ")" + " --> " + self.__class__.__name__ if len(", ".join(var_label_name)) > 0 else self.__class__.__name__ 
            
        else: 
            label = self.__class__.__name__
        
        hash_value.update(label.encode())
        path = Path(self.global_hash_path + hash_value.hexdigest())
        if not path.is_file():
            with path.open(mode='w+') as hash_file:
                hash_file.write(label)
        return self.__class__.__name__ + "_" + "#" + hash_value.hexdigest() if label else self.__class__.__name__



class CLSTask(luigi.Task, LuigiCombinator, CLSBaseTask):
    abstract = True


class CLSWrapperTask(luigi.WrapperTask, LuigiCombinator, CLSBaseTask):
    abstract = True


class CLSExternalTask(luigi.ExternalTask, LuigiCombinator, CLSBaseTask):
    abstract = True
