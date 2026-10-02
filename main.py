"""Entry point for the ACDC data cleaning pipeline.

Run with:
	python main.py

This orchestrates:
	load config -> load raw data -> clean data -> write diagnostics and cleaned data
"""
from src.data import PROJECT_ROOT, load_config, load_data
from src.diagnostics import find_date_order_violations, write_diagnostics
from src.preprocessing import clean_dataset


def main() -> None:
	config = load_config()
	raw_data = load_data(config)

	duplicates_removed = int(raw_data.duplicated().sum())
	cleaned_data, validity_report, invalid_value_details = clean_dataset(raw_data, config)
	date_order_violations = find_date_order_violations(
		cleaned_data,
		config["diagnostics"].get("date_order_rules", []),
		config["diagnostics"].get("id_column"),
	)

	output_path = (PROJECT_ROOT / config["data"]["output_path"]).resolve()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	cleaned_data.to_csv(output_path, sep=config["data"].get("delimiter", ";"), index=False)

	diagnostics_dir = (PROJECT_ROOT / config["diagnostics"]["output_dir"]).resolve()
	write_diagnostics(
		raw_data,
		cleaned_data,
		diagnostics_dir,
		validity_report,
		invalid_value_details,
		date_order_violations,
		config["diagnostics"].get("id_column"),
	)

	print(f"Removed {duplicates_removed} exact duplicate rows.")
	print(f"Found {len(date_order_violations)} end-before-start date pairs; see the diagnostics report.")
	print(f"Saved preprocessed data to {output_path}")
	print(f"Saved diagnostic reports to {diagnostics_dir}")


if __name__ == "__main__":
	main()
