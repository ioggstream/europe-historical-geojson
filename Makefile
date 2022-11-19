

clean:
	rm -f tmp-mappe.yaml*.geojson

save: clean
	poetry run pytest -k save

update:
	poetry run pytest -k save

mappe:
	poetry run python -mmappe
