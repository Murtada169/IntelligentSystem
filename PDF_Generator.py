# title = "2022-09-09 12:00:00 Meeting Minutes"
# filepath = os.path.join('Minutes', '2002-09-09_120000.txt')

# ----- Voon Tao ----- #
from fpdf import FPDF

class PDFGenerator(FPDF):
    
    
    def header(self):
        
        # Arial bold 15
        self.set_font('Arial', 'B', 15)

        # Calculate width of title and position
        with open(filepath) as f:
            title = f.readline().rstrip()

        w = self.get_string_width(title) + 6
        self.set_x((210 - w) / 2)

        # Colors of frame, background and text
        self.set_draw_color(0,0,0)
        self.set_fill_color(65,105,225)
        self.set_text_color(255,255,255)
        
        self.set_line_width(1) # Thickness of frame (1 mm)
        self.cell(w, 9, title, 1, 1, 'C', 1) # Title
        self.ln(10) # Line break

    def footer(self):
        self.set_y(-15) # Position at 1.5 cm from bottom
        self.set_font('Arial', 'I', 8) # Arial italic 8
        self.set_text_color(128) # Text color in gray
        self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C') # Page number
    
    def chapter_body(self):
        f = open(filepath, "r") # Read text file
        self.set_font("Arial", size = 15) # Arial 15
        # Output justified text
        for x in f:
            self.multi_cell(200, 10, txt = x, align = 'L')
        self.ln() # Line break
        self.cell(0, 5, '(end of meeting)')
        
    def print_chapter(self,filepathh):
        global filepath
        filepath = filepathh
        self.add_page()
        self.chapter_body()