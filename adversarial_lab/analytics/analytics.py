from typing import List
from adversarial_lab.db import DB
from adversarial_lab.analytics import Tracker


class AdversarialAnalytics:
    def __init__(self,
                 db: DB,
                 trackers: List[Tracker],
                 table_name: str,
                 force_create_table: bool = False) -> None:
        
        if db is None:
            self.db = None
            self.trackers = []
            self.table_name = None
            return
        
        if not db.validate_connection():
            raise ConnectionError("Failed to Validate DB Connection")
        self.db = db

        if not all(isinstance(tracker, Tracker) for tracker in trackers):
            raise ValueError("All trackers must be of type Tracker")
        self.trackers = trackers

        self.table_name = table_name
        self._initialize(force_create_table)


    def reset_trackers(self) -> None:
        for tracker in self.trackers:
            tracker.reset_values()

    def update_pre_training_values(self,
                             *args,
                             **kwargs) -> None:
        for tracker in self.trackers:
            tracker.pre_training(*args, **kwargs)

    def update_post_batch_values(self,
                                 batch_num: int,
                                 *args,
                                 **kwargs) -> None:
        for tracker in self.trackers:
            tracker.post_batch(batch_num, *args, **kwargs)

    def update_post_epoch_values(self,
                                 epoch_num: int,
                                 *args,
                                 **kwargs) -> None:
        for tracker in self.trackers:
            tracker.post_epoch(epoch_num, *args, **kwargs)

    def update_post_training_values(self,
                                 *args,
                                 **kwargs) -> None:
        for tracker in self.trackers:
            tracker.post_training(*args, **kwargs)

    def _initialize(self, force_create_table: bool) -> None:
        columns = {"epoch_num": "int"}

        for tracker in self.trackers:
            tracker_columns = tracker.get_columns()
            for tracker_column in tracker_columns:
                if tracker_column in columns:
                    raise ValueError(f"Column '{tracker_column}' from '{tracker.__class__.__name__}' already from a previous tracker. Please ensure column names are unique.")
                columns[tracker_column] = tracker_columns[tracker_column]

        self.db.create_table(self.table_name, columns, force=force_create_table)

    def write(self, epoch_num: int) -> None:
        if not self.db:
            return
        
        data = {"epoch_num": epoch_num}

        for tracker in self.trackers:
            tracker_data = tracker.serialize()
            data.update(tracker_data)

        self.db.insert(self.table_name, data)

        self.reset_trackers()