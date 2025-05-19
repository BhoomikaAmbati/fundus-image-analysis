import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from model import build_unet


# Function to capture model summary
def get_model_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    return stream.getvalue()

# Function to save summary as a properly formatted PDF
def save_summary_as_pdf(model, filename="model_summary.pdf"):
    summary_text = get_model_summary(model)

    # Create PDF canvas
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    # Use built-in Courier font (No need for TTF)
    c.setFont("Courier", 8)

    # Margin settings
    x, y = 50, height - 50
    line_height = 10  # Line spacing

    # Write text line by line
    for line in summary_text.split("\n"):
        if y < 50:  # New page if content overflows
            c.showPage()
            c.setFont("Courier", 8)
            y = height - 50
        c.drawString(x, y, line)
        y -= line_height

    # Save the PDF
    c.save()
    print(f"Model summary saved as {filename}")

# Example usage:
test_model, _ = build_unet(input_shape=(512, 512, 3))
save_summary_as_pdf(test_model, "unet_model_summary.pdf")
