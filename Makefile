

clean:
	rm -f tmp-mappe.yaml*.geojson

save: clean
	poetry run pytest -k save
