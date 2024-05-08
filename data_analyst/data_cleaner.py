import numpy as np

class DataCleanerColumn:
    def __init__(self, name, data , index, dtype):
        self.index = index
        self.name = name
        self.data = data
        self.dtype = dtype

    def unique(self,return_counts=False):
        return np.unique(self.data,return_counts=return_counts)
        
    def __str__(self):
        return (f'{self.dtype} with name {self.name} and index {self.index}')

class DataCleaner:

    def __iter__(self):
        return self

    def __next__(self) -> DataCleanerColumn:
        while self._iter_index_type < len(self.supported_dtypes_list):
            #print(self._iter_index_type,self._iter_index_column)
            dtype = self.supported_dtypes_list[self._iter_index_type]
            shp = self._data[dtype][2].shape
            dim = self._data[dtype][2].ndim
            has_any = (shp[0] > 0)
            is_one_dimension_and_zero = (self._iter_index_column == 0)
            is_2d_and_index_within_bounds = ((dim > 1) and (self._iter_index_column < shp[1]))
            if has_any and (is_one_dimension_and_zero or is_2d_and_index_within_bounds):
                col = self[dtype,self._iter_index_column]
                self._iter_index_column += 1
                return col
            #print(self._iter_index_type)
            self._iter_index_column = 0
            self._iter_index_type += 1

        self._iter_index_column = 0
        self._iter_index_type = 0
        raise StopIteration

    def set_default_np_options():
        np.set_printoptions(suppress=True, linewidth=100, precision=2)

    def show_basic_info(self):
        self._iter_index_type = 0
        self._iter_index_column = 0
        for col in self:
            print(col)
                
    def __init__(self,file_path,delimiter,
                 generate=False,
                 restore=False,
                 usecols=None,
                 export_path=None):
        self.supported_dtypes_list = [
            float,
            str,
            np.datetime64,
            int
        ]
        
        self._iter_index_type = 0
        self._iter_index_column = 0

        self.export_path = export_path
        
        self._data = {}
        for dtype in self.supported_dtypes_list:
            self._data[dtype] = [
                                    np.array([],dtype=int),
                                    np.array([],dtype=str),
                                    np.array([],dtype=dtype)
                                ]
            
        self.file_path = file_path
        self.file_delimiter = delimiter
        
        if (generate and restore):
            raise Exception("Both kwargs 'generate' and 'restore' can't be assigned to True")
        if generate:
            self.gen_data(usecols)
        if restore:
            self.restore(file_path)

    def _get_export_name(self,name):
        if self.export_path and (not self.export_path in name):
            name = os.path.join(self.export_path,name)
        return name

    def restore(self,name):
        name = self._get_export_name(name)
        data = np.load(name)
        for dtype in self.supported_dtypes_list:
            base_name = str(dtype).strip().replace(' ','').replace('\'','')
            self._data[dtype][0] = data[base_name + '_column_original_index']
            self._data[dtype][1] = data[base_name + '_header_name']
            self._data[dtype][2] = data[base_name + '_data']
            
    def export(self,name):
        name = self._get_export_name(name)
        dict_arrays = {}
        for dtype in self.supported_dtypes_list:
            base_name = str(dtype).strip().replace(' ','').replace('\'','')
            dict_arrays[base_name + '_column_original_index'] = self._data[dtype][0]
            dict_arrays[base_name + '_header_name'] = self._data[dtype][1]
            dict_arrays[base_name + '_data'] = self._data[dtype][2]
        print(dict_arrays.keys())
        np.savez(name,**dict_arrays)

    def rename_column(self, dtype, index, new_name):
        self._data[dtype][1][index] = new_name
        
    def delete_column(self, dtype, index):
        self._data[dtype][0] = np.delete(self._data[dtype][0],index,axis=None)
        self._data[dtype][1] = np.delete(self._data[dtype][1],index,axis=None)
        self._data[dtype][2] = np.delete(self._data[dtype][2],index,axis=1)

    def add_column(self, dtype, nparray, ori_index, name):
        self._data[dtype][0] = np.append(self._data[dtype][0], ori_index)
        self._data[dtype][1] = np.append(self._data[dtype][1], name)
        try:
            self._data[dtype][2] = np.column_stack([self._data[dtype][2],nparray]).astype(dtype=dtype)
        except Exception as e:
            if self._data[dtype][2].size == 0:
                self._data[dtype][2] = nparray.copy().astype(dtype=dtype)
            else:
                raise e
                
    def move_column(self, from_dtype, from_index, to_dtype):
        ori_index = self._data[from_dtype][0][from_index]
        from_name = self._data[from_dtype][1][from_index]
        from_data = self._data[from_dtype][2][:,from_index]

        self.add_column(to_dtype, from_data, ori_index, from_name)
        self.delete_column(from_dtype,from_index)

    def try_datetime(self, nparray, datetime_format):
        # datetime lib handles more format options like:
        # '%b-%y' (Mmm-YY : May-15)
        unique_dates = np.unique(nparray)
        array_dates = [np.datetime64('NaT') 
                       if date_str == '' else np.datetime64(datetime.strptime(date_str,datetime_format)) 
                       for date_str in unique_dates]
        cpy = nparray.copy()
        for i in range(unique_dates.shape[0]):
            cpy[cpy == unique_dates[i]] = array_dates[i]
        # now is safe to overwrite
        nparray[:] = cpy[:]
        return cpy

    def get_type_data(self, dtype):
        return self._data[dtype][2]
        
    def __getitem__(self, key) -> DataCleanerColumn:
        dtype,index = key
        name = self._data[dtype][1][index]
        try:
            data = self._data[dtype][2][:,index]
        except IndexError as e:
            data = self._data[dtype][2][:]
        return DataCleanerColumn(name, data, index, dtype)
    
    def gen_data(self,usecols=None):
        data_nan = np.genfromtxt(self.file_path,
                                 delimiter=self.file_delimiter,
                                 skip_header=True,
                                 usecols=usecols)
        # first pass
        # open only numeric data
        # calculate max val as placeholder for NaN
        # identify non-numeric columns (where calculated mean is NaN)
        temporary_max_val_plus_one = np.nanmax(data_nan) + 1
        temp_mean = np.atleast_1d(np.nanmean(data_nan,axis=0))
        
        self.temp_stats = np.array([np.atleast_1d(np.nanmin(data_nan,axis = 0)),
                                  temp_mean,
                                  np.atleast_1d(np.nanmax(data_nan, axis = 0))])

        self._data[str][0] = np.atleast_1d(np.argwhere(np.isnan(temp_mean)).squeeze())
        self._data[float][0] = np.atleast_1d(np.argwhere(~np.isnan(temp_mean)).squeeze())

        npa_usecols = np.atleast_1d(usecols)
        if usecols is None:
            npa_usecols = np.atleast_1d(np.arange(0, data_nan.shape[0]))
            
        # second pass
        # STR
        if self._data[str][0].shape[0] > 0:
            # get source index of columns if usecols is passed
            self._data[str][0] = npa_usecols[self._data[str][0]]
            self._data[str][2] = np.genfromtxt(self.file_path,
                                          delimiter=self.file_delimiter,
                                          dtype=str,
                                          usecols=self._data[str][0],
                                          skip_header=True)

        # FLOAT
        if self._data[float][0].shape[0] > 0:
            # get source index of columns if usecols is passed
            self._data[float][0] = npa_usecols[self._data[float][0]]
            self._data[float][2] = np.genfromtxt(self.file_path,
                                          dtype='float',
                                          delimiter=self.file_delimiter,
                                          usecols=self._data[float][0],
                                          skip_header=True,
                                          filling_values=temporary_max_val_plus_one)

        # this works because the above data contains all rows minus header
        skip_footer_count = max(self._data[str][2].shape[0], self._data[float][2].shape[0])
        headers_all = np.genfromtxt(self.file_path,
                                    delimiter=self.file_delimiter,
                                    dtype=str,
                                    skip_footer=skip_footer_count)
        
        self._data[str][1] = headers_all[self._data[str][0]]
        self._data[float][1]= headers_all[self._data[float][0]]

    def __str__(self):
        output = ''
        for dtype in self.supported_dtypes_list:
            output += f'\nSUMMARY FOR {str(dtype)}'
            output += f'\nColumn original index:\n{self._data[dtype][0]}'
            output += f'\nColumn name:\n{self._data[dtype][1]}'
            output += f'\nData:\n{self._data[dtype][2]}\n'
        return output
