# encoding=UTF-8

"""Load, store, append and analyze experimental results.

Modules used when trying out inputs in a parameter space: when there is a lot
of parameters to adjust, keeping track of the results we obtain is painful.
The class in this module aims at keeping track of those results and the
corresponding parameters in a smooth and transparent way.

Results can be easily appended to existing ones (using the DataManager behind
the scenes), and then retrieved and filtered.
"""

# project
import datamanager

class Results(object):
    """Defines a generic interface for results storage and retrieval."""

    def __init__(self, path):
        """Initializes the Results object.

        Args:
            path: the DataManager path where the data is
        """
        self.path = path

    def add_result(self, params, results, metadata={}):
        """Add some results to storage.

        Args:
            params: the parameters used
            results: the results obtained
            metadata: a dictionary containing metadata about the results.
                Existing metadata will be merged with what is provided (but only
                within limits, ie key collisions will be overwritten with the
                provided values)
        """
        dm = datamanager.DataManager(self.path)

        meta, data = {}, {'data': []}
        if dm.exists():
            meta, data = dm.get()

        # Add/overwrite the existing metadata
        for k, v in metadata.iteritems():
            meta[k] = v

        data['data'].append((params, results))
        dm.set(metadata=meta, data=data)

    def get_results(self, pf={}, rf={}):
        """Returns results, filtered as specified.

        Args:
            pf: a dictionary, where each key is of the form 'some.param.path',
                and a value v. The available data will be filtered on the values
                of params['some']['param']['path']: only results where this is
                equal to v are kept.
                All filter items must match for a (params, result) item to be kept.
                A value of None means the parameter must NOT be set, or equal to
                None.
            rf: an iterable of paths of the the same form as pf, which filters
                what will be included in the returned list of results

        Returns:
            A (meta, data) couple, where data is a [(params, result)] list,
            filtered as specified.
        """
        dm = datamanager.DataManager(self.path)
        if not dm.exists():
            return ({}, [])
        meta, data = dm.get()

        filtered_items = []
        for params, result in data['data']:
            add = True
            for p_path, p_value in pf.iteritems():
                if not self._is_path_equal(params, p_path, p_value):
                    add = False
                    break
            if not add:
                continue

            filtered_items.append(
                (params, self._filter_result(result, rf))
            )

        return meta, filtered_items

    def _is_path_equal(self, params, parampath, value):
        """Checks if a parameter path exists and has the given value.

        Args:
            params: the parameter dictionary
            parampath: the path in the parameters dict, as a string of dot ('.')
                separated key names
            value: the expected value. If None, we do not want to find a value
                (although finding None is fine)

        Returns:
            True if params[parampath[0]][...][parampath[-1]] == v.
        """
        d = params
        for k in parampath.split('.'):
            if k not in d:
                if value is None:
                    return True
                return False
            d = d[k]

        return d == value

    def _filter_result(self, result, f):
        """Filters the items of a result, to select only certain keys.

        Args:
            result: dictionary containing the result data
            f: iterable containing the keys to select. If empty, everything is
                returned

        Returns:
            A dictionary containing only the data specified by the elements of f.
        """
        if not f:
            return result

        # New result dictionary
        d = {}
        for path in f:
            result_tmp = result
            d_tmp = d
            path_list = path.split('.')
            for i, key in enumerate(path_list):
                # If we're on the last item
                if i == len(path_list)-1:
                    d_tmp[key] = result_tmp.get(key)
                else:
                    result_tmp = result_tmp.get(key, {})
                    d_tmp[key] = d_tmp.get(key, {})
                    d_tmp = d_tmp[key]

        return d

