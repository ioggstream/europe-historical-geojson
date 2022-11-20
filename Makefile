

clean:
	rm -f tmp-mappe.yaml*.geojson
	find .  -depth -path '__pycache_*'_  -delete

save: clean
	poetry run pytest -k save

update:
	poetry run pytest -k save

mappe:
	poetry run python -mmappe
