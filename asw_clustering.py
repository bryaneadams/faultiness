import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform, pdist
from scipy.cluster import hierarchy

from typing import List, Union, Tuple
from numpy.typing import ArrayLike


class Asw_clustering:
    def __init__(self) -> None:
        pass

    def create_numerical_df(
        self, df: pd.DataFrame, dummy_columns: list
    ) -> pd.DataFrame:
        """Creates an encoded dataframe for the use in the ASW clustering method.

        Args:
            df (pd.DataFrame): data you want to cluster
            dummy_columns (list): columns you want encoded. All other columns will remain the same

        Returns:
            pd.DataFrame: the data frame with encodings for specified columns
        """

        subset_df = df[dummy_columns]
        keep_df = df.drop(columns=dummy_columns)
        dummy_df = pd.get_dummies(subset_df, dtype=float) * 1 / (2 ** (1 / 2))

        return pd.concat([keep_df, dummy_df], axis=1)

    def _h_clustering(
        self, data: pd.DataFrame, n_cluster: int, method: str = "ward"
    ) -> ArrayLike:
        """Conduct hierarchical clustering based on provided arguments

        Args:
            data (pd.DataFrame): data to be clustered
            n_cluster (int): number of clusters
            method (str, optional): Method to use. Defaults to 'ward'.

        Returns:
            ArrayLike: initial cluster assignments
        """

        linkage = hierarchy.linkage(data, method=method)
        cluster_assignments = hierarchy.cut_tree(linkage, n_clusters=n_cluster)

        return cluster_assignments.T[0]

    def _dist_matrix_e(
        self, data: Union[ArrayLike, pd.DataFrame], weights: ArrayLike = None
    ) -> ArrayLike:
        """Calculate n x n Euclidean distance matrix, containing the pairwise Euclidean
        distances between the n group members

        Args:
            data (Union[ArrayLike,pd.DataFrame]): dummy encoded data
            weights (ArrayLike, optional): Weights for features. Defaults to None.

        Returns:
            ArrayLike: n x n Euclidean distance matrix
        """
        # Check if the input is a DataFrame; if so, convert to a NumPy array
        if isinstance(data, pd.DataFrame):
            data = data.values

        # If weights are not provided, set them to 1 for all columns
        if weights is None:
            weights = np.ones(data.shape[1])
        else:
            # Ensure weights are a NumPy array for consistency
            weights = np.asarray(weights)

        # Apply weights to data
        weighted_data = data * weights

        # Calculate pairwise distances using pdist and convert to a square matrix
        # The 'euclidean' metric is used because we're calculating the Euclidean distance
        dist_matrix = squareform(pdist(weighted_data, metric="euclidean"))

        return dist_matrix

    def _calculate_asw(self, dist_matrix: ArrayLike, assignments: ArrayLike) -> float:
        """Calculates the average silhouette width for cluster assignments

        Args:
            dist_matrix (ArrayLike): distance matrix, n x n
            assignments (ArrayLike): cluster assignments 1 x n

        Returns:
            float: the average silhouette width for current assignments
        """
        a_i = np.zeros(len(assignments))
        b_i = np.zeros(len(assignments))
        classes = np.unique(assignments)
        num_classes = len(classes)

        # If there in only one class in current assignments
        if num_classes == 1:
            return 0

        isolates = []

        for idx, c_l in enumerate(assignments):

            all_b_i = np.zeros(num_classes)

            # Find neighbors in their assigned cluster
            g = np.where(assignments == c_l)[0]
            neighbors = g[g != idx]

            # if they are an isolate, their sil width becomes 0
            if len(neighbors) == 0:
                isolates.append(idx)
                a_distances = [0]
            # get distances for neighbors
            else:
                a_distances = dist_matrix[idx][neighbors]

            # get distances for members of other clusters
            for c in classes:
                if c != c_l:

                    g = np.where(assignments == c)[0]

                    b_distances = dist_matrix[idx][g]

                    index = np.where(classes == c)[0][0]
                    all_b_i[index] = np.mean(b_distances)

            # remove the 0 for the assigned group
            c_l_index = np.where(classes == c_l)[0][0]
            all_b_i = np.delete(all_b_i, c_l_index)

            a_i[idx] = np.mean(a_distances)
            b_i[idx] = np.min(all_b_i)

        sil_width = (b_i - a_i) / np.maximum(a_i, b_i)

        # if a person is by themselves, their sil width becomes 0
        if len(isolates) > 0:
            for i in isolates:
                sil_width[i] = 0

        asw = np.mean(sil_width)

        return asw

    def _reassign_members(
        self, dist_matrix: ArrayLike, assignments: ArrayLike
    ) -> Tuple[dict, float, ArrayLike]:
        """Iterates through each member and then assigns them to different
        clusters. For each iteration every member is assigned to the other cluster.
        The optimal move is the maximum ASW calculated starting with the initial assignments.

        Args:
            dist_matrix (ArrayLike): distances
            assignments (ArrayLike): cluster assignments

        Returns:
            Tuple(dict,float,ArrayLike): dictionary of optimal change id: group, maximum ASW, optimal cluster assignments
        """

        classes = np.unique(assignments)
        update_move = {}
        optimal_assignment = assignments.copy()
        max_asw = self._calculate_asw(dist_matrix, assignments)

        for idx, c_l in enumerate(assignments):

            for new_class in classes:
                if c_l != new_class:
                    temp_assignment = assignments.copy()
                    temp_assignment[idx] = new_class

                    new_asw = self._calculate_asw(dist_matrix, temp_assignment)

                    if new_asw > max_asw:
                        update_move = {idx: new_class}
                        optimal_assignment = temp_assignment.copy()
                        max_asw = new_asw

        return update_move, max_asw, optimal_assignment

    def transform_data(self, df: pd.DataFrame, fields: list) -> pd.DataFrame:
        """Transforms your data so you can use the clustering methods

        Args:
            df (pd.DataFrame): data to cluster
            fields (list): List of fields to encode with 0, 1/sqrt(2)

        Returns:
            pd.DataFrame: Transformed data
        """
        df = df.copy()
        dummy_df = self.create_numerical_df(df, fields)

        return dummy_df

    def _fit_by_team(
        self,
        data: pd.DataFrame,
        n_clusters: int,
        team_assignment: str,
        quiet: bool = True,
    ) -> pd.DataFrame:
        """Used when you want to cluster by many teams in one table

        Args:
            data (pd.DataFrame): data to cluster
            n_clusters (int): maximum number of clusters
            team_assignment (str): field with current team assignment
            quiet (bool, optional): Used to output the iterations. Defaults to True.

        Returns:
            pd.DataFrame: table of cluster results by team id
        """

        team_ids = np.unique(data[team_assignment].to_numpy())
        result_dict = {}

        for team_id in team_ids:
            result_dict[team_id] = self.fit(
                data[data[team_assignment] == team_id], n_clusters, quiet=quiet
            )

        return_df = pd.DataFrame(result_dict).T
        return_df["team_id"] = team_ids

        return return_df[
            ["team_id", "asw", "optimal_assignment", "number_of_groups", "method"]
        ]

    def fit(
        self,
        data: pd.DataFrame,
        n_clusters: int,
        team_assignment: str = None,
        quiet: bool = True,
        return_type: str = "dict",
    ) -> Union[dict, pd.DataFrame]:
        """Executes the algorithm by first running Ward followed by Average

        #TODO add weights

        Args:
            data (pd.DataFrame): data to fit
            n_clusters (int): max number of clusters
            team_assignment (str, optional): name a field with current assignments. Defaults to None.
            quiet (bool, optional): Used to output the iterations. defaults to False. Defaults to True.
            return_type (str, optional): "df" or "dict" for return data structure. Defaults to "dict".

        Raises:
            ValueError: raises an error if your number of clusters exceeds the number of observations

        Returns:
            Union[dict,pd.DataFrame]: results of cluster
        """

        if n_clusters > data.shape[0]:
            raise ValueError(
                "The number of clusters {} exceeds the number of observations {}".format(
                    n_clusters, data.shape[0]
                )
            )

        if team_assignment:
            return self._fit_by_team(data, n_clusters, team_assignment, quiet)

        dist_matrix = self._dist_matrix_e(data)

        best_fit = {
            "asw": -99,
            "optimal_assignment": [],
            "number_of_groups": 0,
            "method": "",
        }

        for n in range(2, n_clusters + 1):

            initial_assignments = self._h_clustering(data, n, "ward")
            initial_asw = self._calculate_asw(dist_matrix, initial_assignments)

            update_move, temp_max_asw, optimal_assignments = self._reassign_members(
                dist_matrix, initial_assignments
            )

            if temp_max_asw > best_fit["asw"]:
                best_fit["optimal_assignment"] = optimal_assignments
                best_fit["asw"] = temp_max_asw
                best_fit["method"] = "ward"
                best_fit["number_of_groups"] = n

            if not quiet:
                if initial_asw > temp_max_asw:
                    print(
                        "Method:{}, groups:{}, AWS:{}: Assignments:{}".format(
                            "ward", n, round(initial_asw, 6), initial_assignments
                        )
                    )
                else:
                    print(
                        "Method:{}, groups:{}, AWS:{}: Assignments:{} -> {}".format(
                            "ward",
                            n,
                            round(temp_max_asw, 6),
                            initial_assignments,
                            optimal_assignments,
                        )
                    )

        for n in range(2, n_clusters + 1):

            initial_assignments = self._h_clustering(data, n, "average")

            update_move, temp_max_asw, optimal_assignments = self._reassign_members(
                dist_matrix, initial_assignments
            )

            if temp_max_asw > best_fit["asw"]:
                best_fit["optimal_assignment"] = optimal_assignments
                best_fit["asw"] = temp_max_asw
                best_fit["method"] = "average"
                best_fit["number_of_groups"] = n
            elif temp_max_asw == best_fit["asw"] and best_fit["method"] == "ward":
                best_fit["method"] = ["ward", "average"]

            if not quiet:
                if initial_asw > temp_max_asw:
                    print(
                        "Method:{}, groups:{}, AWS:{}: Assignments:{}".format(
                            "average", n, round(initial_asw, 6), initial_assignments
                        )
                    )
                else:
                    print(
                        "Method:{}, groups:{}, AWS:{}: Assignments:{} -> {}".format(
                            "average",
                            n,
                            round(temp_max_asw, 6),
                            initial_assignments,
                            optimal_assignments,
                        )
                    )
        if return_type == "df":
            subgroup_sizes = np.unique(
                best_fit["optimal_assignment"], return_counts=True
            )[1]
            best_fit["subgroup_sizes"] = subgroup_sizes
            return pd.DataFrame([best_fit])

        return best_fit

    def calculate_faultline_value(
        self,
        data: pd.DataFrame,
        group_assignments: ArrayLike,
        team_assignments: Union[list, ArrayLike],
    ) -> pd.DataFrame:
        """Not in use

        Args:
            data (pd.DataFrame): _description_
            group_assignments (ArrayLike): _description_
            team_assignments (Union[list, ArrayLike]): _description_

        Returns:
            pd.DataFrame: _description_
        """

        unique_teams = np.unique(team_assignments)
        data["team_id"] = team_assignments
        data["group_assignments"] = group_assignments

        return_df = pd.DataFrame(
            columns=[
                "fautline_value",
                "assignments",
                "num_sub_groups",
                "sub_group_sizes",
            ],
            dtype=object,
        )

        for g in unique_teams:
            temp_df = data[data["team_id"] == g]
            dist_matrix = self._dist_matrix_e(temp_df)
            assignments = temp_df["group_assignments"].to_numpy()

            asw = self._calculate_asw(dist_matrix, assignments)
            print(asw)
            return_df.at[g, "fautline_value"] = asw
            return_df.at[g, "assignments"] = assignments
            return_df.at[g, "num_sub_groups"] = len(assignments)
            return_df.at[g, "sub_group_sizes"] = np.unique(
                assignments, return_counts=True
            )[1]

        return return_df
