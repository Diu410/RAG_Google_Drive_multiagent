"""Bounded arithmetic and explicit UTC normalization; no inferred source facts.

These hints never invent an event, date, row filter or participant. Citation
selection and independent semantic checks still bind the operation to evidence.
"""
from __future__ import annotations
import ast
import csv
from datetime import date
from fractions import Fraction
import io
from itertools import combinations
import re

_UTC = re.compile(r'(?<![\d:])(?P<hour>\d{1,2}):(?P<minute>\d{2})\s*(?:по\s+)?UTC'
                  r'(?:(?P<sign>[+−-])(?P<oh>\d{1,2})(?::(?P<om>\d{2}))?)?(?![\w:+−-])', re.I)
_NUM = r'\d+(?:[.,]\d+)?'
_EXPRESSION = re.compile(r'(?<![\w.:])(?P<left>-?' + _NUM + r'(?:\s*[+*/×÷−-]\s*-?' + _NUM +
                         r'){1,7})\s*=\s*(?P<right>-?' + _NUM + r')(?![\w:]|[.,]\d)')


def utc_normalizations(text):
    rows, seen = [], set()
    for match in _UTC.finditer(text):
        literal = match.group()
        if literal in seen:continue
        hour, minute = int(match['hour']), int(match['minute'])
        offset_hour, offset_minute = int(match['oh'] or 0), int(match['om'] or 0)
        if hour > 23 or minute > 59 or offset_hour > 14 or offset_minute > 59 or offset_hour == 14 and offset_minute:
            continue
        offset = (offset_hour * 60 + offset_minute) * (-1 if match['sign'] in {'-', '−'} else 1)
        utc_minutes = hour * 60 + minute - offset
        day, clock = divmod(utc_minutes, 24 * 60)
        rows.append({'literal':literal,'utc':f'{clock // 60:02d}:{clock % 60:02d} UTC',
                     'day_shift':day})
        seen.add(literal)
        if len(rows) >= 16:break
    return rows


def _number_expression(text):
    if len(text) > 120:raise ValueError('Arithmetic expression too long')
    node = ast.parse(text.replace(',', '.').replace('×','*').replace('÷','/').replace('−','-'),mode='eval')
    if sum(1 for _ in ast.walk(node)) > 40:raise ValueError('Too many arithmetic nodes')
    def evaluate(item):
        if isinstance(item,ast.Expression):return evaluate(item.body)
        if isinstance(item,ast.Constant) and type(item.value) in (int,float):
            value=Fraction(ast.get_source_segment(text.replace(',', '.').replace('×','*').replace('÷','/').replace('−','-'),item))
            if abs(value)>10**18:raise ValueError('Arithmetic number too large')
            return value
        if isinstance(item,ast.UnaryOp) and isinstance(item.op,(ast.UAdd,ast.USub)):
            return evaluate(item.operand) * (-1 if isinstance(item.op,ast.USub) else 1)
        if isinstance(item,ast.BinOp) and isinstance(item.op,(ast.Add,ast.Sub,ast.Mult,ast.Div)):
            left,right=evaluate(item.left),evaluate(item.right)
            if isinstance(item.op,ast.Add):return left+right
            if isinstance(item.op,ast.Sub):return left-right
            if isinstance(item.op,ast.Mult):return left*right
            return left/right
        raise ValueError('Only bounded numeric arithmetic is allowed')
    return evaluate(node)


def arithmetic_checks(text):
    rows=[]
    for match in _EXPRESSION.finditer(text.replace('`','').replace('**','')):
        try:
            left,right=_number_expression(match['left']),_number_expression(match['right'])
        except (ValueError,SyntaxError,ZeroDivisionError,OverflowError):continue
        rows.append({'expression':match.group(),'correct':left==right,
                     'result':str(left.numerator) if left.denominator==1 else str(left)})
        if len(rows)>=12:break
    return rows


def source_calculations(sources):
    output=[]
    for source in sources:
        hints=calculation_hints(source['text'])
        if hints:output.append({'chunk_id':source['chunk_id'],**hints})
    return output


def table_sums(text):
    """Literal delimited tables only; label each numeric column and group.

    A group never implies its relevance to the question. Single-row groups add
    no new arithmetic; they are already explicit in the original row.
    """
    lines=text.strip().splitlines()
    if not 3<=len(lines)<=201:return []
    delimiter=max((',', ';', '\t'),key=lines[0].count)
    if delimiter not in lines[0]:return []
    try:rows=list(csv.reader(io.StringIO(text),delimiter=delimiter))
    except csv.Error:return []
    if not rows or not 2<=len(rows[0])<=8:return []
    header,data=rows[0],rows[1:]
    if any(len(row)!=len(header) for row in data) or len(set(header))!=len(header):return []
    numeric=[]
    for index in range(len(header)):
        if all(re.fullmatch(r'-?\d{1,15}(?:[.,]\d{1,8})?',row[index].strip()) for row in data):numeric.append(index)
    categorical=[i for i in range(len(header)) if i not in numeric]
    output=[]
    for column in numeric:
        for size in (1,2):
            for keys in combinations(categorical,size):
                groups={}
                for number,row in enumerate(data,2):
                    group=tuple(row[key] for key in keys)
                    groups.setdefault(group,[]).append((number,row[column]))
                for group,values in groups.items():
                    if len(values)<2:continue
                    total=sum((Fraction(value.replace(',','.')) for _,value in values),Fraction(0))
                    output.append({'column':header[column],'group':dict(zip((header[k] for k in keys),group)),
                        'source_lines':[n for n,_ in values],'sum':str(total.numerator) if total.denominator==1 else str(total)})
                    if len(output)>=32:return output
    return output


def calculation_hints(text):
    output={}
    times=utc_normalizations(text)
    if times:
        output['utc_normalizations']=times
        pairs=[]
        for a,b in combinations(times[:8],2):
            def minutes(item):
                hour,minute=map(int,item['utc'].split()[0].split(':'))
                return item['day_shift']*1440+hour*60+minute
            delta=minutes(b)-minutes(a)
            pairs.append({'earlier_if_same_source_date':a['literal'] if delta>=0 else b['literal'],
                'later_if_same_source_date':b['literal'] if delta>=0 else a['literal'],
                'difference_minutes_if_same_source_date':abs(delta)})
        if pairs:output['utc_comparisons']=pairs
    sums=table_sums(text)
    if sums:output['table_sums']=sums
    return output


_DATE = re.compile(r'(?<!\w)(?:\d{4}-\d{2}-\d{2}|\d{1,2}[./]\d{1,2}[./]\d{4}|'
    r'\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+\d{4})(?!\w)',re.I)
_SHARED_DAY = re.compile(r'\b(?:все|обе|оба)\s+(?:записи|события|метки|отметки)\s+'
    r'(?:относятся\s+к|отнесены\s+к|датированы|произошли(?:\s+в)?)\s+'
    +_DATE.pattern+r'(?:\s+года)?(?=\s*(?:[.!?\n]|$))',re.I)
_RELATIVE_DATE = re.compile(r'\b(?:вчера|завтра|позавчера|послезавтра|кроме|исключени\w*|'
    r'(?:следующ\w*|предыдущ\w*|друг\w*)\s+(?:день|дня|дате?|сут\w*))\b',re.I)


def _valid_date(literal):
    months = ('января','февраля','марта','апреля','мая','июня','июля','августа',
              'сентября','октября','ноября','декабря')
    try:
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}',literal):
            date.fromisoformat(literal)
        elif re.fullmatch(r'\d{1,2}[./]\d{1,2}[./]\d{4}',literal):
            day,month,year=map(int,re.split(r'[./]',literal))
            date(year,month,day)
        else:
            day,month,year=literal.casefold().split()
            date(int(year),months.index(month)+1,int(day))
        return True
    except (ValueError,TypeError):return False


def source_time_comparison(question, sources):
    """Exact calculator for two explicitly dated UTC stamps, not a world claim.

    Refuse ambiguity: one source, exactly two distinct stamps, an explicit
    shared-date statement and a mentioned identifier or filename. Other temporal
    questions retain the normal source-based generation and independent audit.
    """
    if not re.search(r'\b(?:раньше|позже|разниц\w*\s+(?:во\s+)?времен\w*)\b',question,re.I):return None
    identifiers=set(re.findall(r'(?<!\w)[A-Za-zА-Яа-я]+[-_]\d+(?!\w)',question.casefold()))
    eligible=[]
    for source in sources:
        text=source['text']
        path=str((source.get('metadata') or {}).get('source',''))
        def has_id(value,identifier):
            return bool(re.search(r'(?<!\w)'+re.escape(identifier)+r'(?!\w)',value,re.I))
        named_file=bool(path and path.casefold() in question.casefold())
        if not (identifiers and all(has_id(text,identifier) for identifier in identifiers) or named_file):continue
        shared=_SHARED_DAY.search(text)
        dates={m.group().casefold() for m in _DATE.finditer(text)}
        if not shared or len(dates)!=1 or not _DATE.search(shared.group()):continue
        if _RELATIVE_DATE.search(text):continue
        if not _valid_date(next(iter(dates))):continue
        stamps=list(_UTC.finditer(text))
        normalized=utc_normalizations(text)
        if len(stamps)!=2 or len(normalized)!=2:continue
        items=[]
        for stamp,norm in zip(stamps,normalized):
            start=text.rfind('\n',0,stamp.start())+1
            end=text.find('\n',stamp.end())
            end=len(text) if end<0 else end
            literal_line=text[start:end].strip()
            if not literal_line or len(literal_line)>240:break
            if identifiers and not named_file and not any(has_id(literal_line,i) for i in identifiers):break
            h,m=map(int,norm['utc'].split()[0].split(':'))
            items.append({'quote':literal_line,**norm,'utc_minutes':norm['day_shift']*1440+h*60+m})
        if len(items)==2:eligible.append((source,shared.group(),items))
    if len(eligible)!=1:return None
    source,date_witness,items=eligible[0]
    earlier,later=sorted(items,key=lambda row:row['utc_minutes'])
    difference=later['utc_minutes']-earlier['utc_minutes']
    path=str(source.get('metadata',{}).get('source',source['chunk_id']))
    def clock(item):
        shift=item['day_shift']
        return item['utc']+(f' (сдвиг суток {shift:+d})' if shift else '')
    if difference:
        answer=f"Раньше: «{earlier['quote']}» — {clock(earlier)}.\n\nПозже: «{later['quote']}» — {clock(later)}."
    else:
        answer=f"Метки совпадают по UTC: «{earlier['quote']}» и «{later['quote']}» — {clock(earlier)}."
    answer=f'Расчёт при общей дате, указанной в источнике: «{date_witness}».\n\n'+answer
    answer+=f"\n\nРазница: {difference} минут.\n\nИсточник: `{path}`."
    return {'answer':answer,'source':source,'date_witness':date_witness,'earlier':earlier,'later':later,
            'difference_minutes':difference,'method':'explicit_shared_date_utc_comparison'}
