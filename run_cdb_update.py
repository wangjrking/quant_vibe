from data_process_module import download_cdb_data


if __name__ == "__main__":
    df = download_cdb_data(down_db=True, data_file_url="../data_file")
    print("download_cdb_done", df.shape, df["trade_date"].astype(str).max(), flush=True)
