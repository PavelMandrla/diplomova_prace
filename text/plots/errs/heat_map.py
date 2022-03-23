import svgwrite

dwg = svgwrite.Drawing(height='1000px', width='1000px')
dwg.add(dwg.line((0, 0), (10, 50), stroke=svgwrite.rgb(255, 0, 0, '%')))
dwg.add(dwg.text('Test', insert=(0, 10), fill='blue'))
dwg.add(dwg.rect((0, 0), (10, 10), fill='blue'))

dwg.save()