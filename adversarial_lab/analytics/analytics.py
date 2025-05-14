import warnings

from adversarial_lab.db import DB
from adversarial_lab.analytics import Tracker

from typing import List, Optional


class AdversarialAnalytics:
    def __init__(self,
                 db: Optional[DB] = None,
                 trackers: Optional[List[Tracker]] = None,
                 table_name: Optional[str] = None,
                 force_create_table: bool = False) -> None:
        """
        Tracks various metrics during the attack process.

        Initializes the database connection, trackers, and creates a table with the necessary columns.
        If ``db`` is ``None``, the class will not track any metrics.

        Parameters
        ----------
        db : DB, optional
            The database object to use for storing the tracked data.
        trackers : list of Tracker, optional
            A list of Tracker objects to track various metrics.
        table_name : str, optional
            The name of the table to store the tracked data.
        force_create_table : bool, optional
            If True, the table will be deleted and re-created.

        Raises
        ------
        ConnectionError
            If unable to connect to the database.
        TypeError
            If the trackers are not of type ``Tracker``.
        ValueError
            If the table name is not provided.
        """

        if db is None:
            self.db = None
            self.trackers = []
            self.table_name = None
            return
        
        if table_name is None:
            raise ValueError("Table name must be provided.")

        if not db.validate_connection():
            raise ConnectionError("Failed to Validate DB Connection")
        self.db = db

        if not all(isinstance(tracker, Tracker) for tracker in trackers):
            raise TypeError("All trackers must be of type 'Tracker'")
        self.trackers = trackers

        self.table_name = table_name
        self._initialize(force_create_table)

        self.warned = {}

    def reset_trackers(self) -> None:
        """
        Resets the values of all trackers. Called at the beginning of each epoch to reset the values of all trackers.
        """
        for tracker in self.trackers:
            tracker.reset_values()

    def update_pre_attack_values(self,
                                 *args,
                                 **kwargs) -> None:
        """
        Updates the values of all trackers before staring attack. Tracks data befor the attack process begins.
        """
        for tracker in self.trackers:
            tracker.pre_attack(*args, **kwargs)

    def update_post_batch_values(self,
                                 batch_num: int,
                                 *args,
                                 **kwargs) -> None:
        """
        Updates the values of all trackers after each batch if track_batch is True. Called after each batch during the attack process.
        """
        for tracker in self.trackers:
            tracker.post_batch(batch_num, *args, **kwargs)

    def update_post_epoch_values(self,
                                 *args,
                                 **kwargs) -> None:
        """
        Updates the values of all trackers after each epoch if track_epoch is True. Called after each epoch during the attack process.
        """
        for tracker in self.trackers:
            tracker.post_epoch(*args, **kwargs)

    def update_post_attack_values(self,
                                  *args,
                                  **kwargs) -> None:
        """
        Updates the values of all trackers after the attack. Called after the attack process ends."""
        for tracker in self.trackers:
            tracker.post_attack(*args, **kwargs)

    def _initialize(self, force_create_table: bool) -> None:
        """
        Initializes the database connection, trackers, and creates a table with the necessary columns.

        Args:
            force_create_table (bool, optional): If True, the table will be deleted and re-created.

        Raises:
            ValueError: If the table name is not provided.
        """
        columns = {"epoch_num": "int"}

        for tracker in self.trackers:
            tracker_columns = list(tracker.serialize().keys())
            for tracker_column in tracker_columns:
                if tracker_column in columns:
                    raise ValueError(
                        f"Column '{tracker_column}' from '{tracker.__class__.__name__}' already from a previous tracker. Please ensure column names are unique.")
                columns[tracker_column] = tracker._columns[tracker_column]

        self.db.create_table(table_name=self.table_name,
                             schema=columns,
                             force=force_create_table)

    def write(self,
              epoch_num: int
              ) -> None:
        """
        Writes the tracked data to the database for an epoch.

        Args:
            epoch_num (int): The epoch number to write the data for.

        Raises:
            ConnectionError: If unable to connect to the database or has connection issues.
        """
        if not self.db:
            return

        try:
            data = {"epoch_num": epoch_num}

            for tracker in self.trackers:
                tracker_data = tracker.serialize()
                data.update(tracker_data)

            self.db.insert(self.table_name, data)
            self.reset_trackers()
        except ConnectionError as e:
            if self.warned.get("connection_error", False):
                return
            warnings.warn(f"Failed to connect to the database: {e}")
            self.warned["connection_error"] = True
