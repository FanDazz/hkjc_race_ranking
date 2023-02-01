from datetime import date, datetime, timedelta


def parse_date(input_date:datetime.date, strf=False):
    """ Transform a input datetime variable into expected form.
        Here '%Y/%m/%d' would be the form taken by hkjc for date retrieving.
    """
    if strf:
        return input_date.strftime("%Y/%m/%d")
    return input_date


def init_date(from_date):
    """ Initialize the date of beginning
        , by parsing the input, or use the running date as init.
    """
    if from_date is None:
        from_date = date.today()
    elif isinstance(from_date, str):
        from_date = datetime.strptime(from_date, "%Y/%m/%d")

    return from_date


def get_last_DoW(from_date=None, strf=False, date_of_interest='wednesday'):
    """ Compute last date of week(DoW) we are interested in.
        If from_date is None, we'll retrieve the last DoW from the execution day.

        :param from_date: datetime.date, str
            the beginning date

        :param strf: Bool
            if we are interested in string format

        :return: last date of interest

    """
    assert date_of_interest.lower() in ('wednesday', 'sunday', 'wed', 'sun')

    from calendar import WEDNESDAY, SUNDAY

    DoW = WEDNESDAY if date_of_interest in ('wednesday', 'wed') else SUNDAY
    from_date = init_date(from_date)
    offset = (from_date.weekday() - DoW) % 7
    last_wednesday = from_date - timedelta(days=offset)

    return parse_date(last_wednesday, strf)


def get_span_of_DoW(begin_date, end_date=None, span=None, strf=True, date_of_interest='wednesday'):
    """ Compute a list of weekday given date span or date range.
        Note that end date should be smaller than begin date.
    """ 
    begin_date = get_last_DoW(begin_date, date_of_interest=date_of_interest)

    datelist = []
    # if consider span
    if span is not None:
        for i in range(span):
            last_DoW = begin_date - timedelta(days=7*i)
            datelist.append(parse_date(input_date=last_DoW, strf=strf))

    # if consider date range
    elif end_date is not None:
        end_date = parse_date(get_last_DoW(end_date, date_of_interest=date_of_interest), strf=strf)

        for i in range(1000):
            last_DoW = begin_date - timedelta(days=7*i)
            parsed_last_DoW = parse_date(input_date=last_DoW, strf=strf)

            if parsed_last_DoW==end_date:
                break
            else:
                datelist.append(parsed_last_DoW)

    return datelist