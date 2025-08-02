from sqlalchemy import Column, DateTime, BigInteger, Float, Integer, Index, Boolean, ForeignKey, String
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker, backref, relation
from sqlalchemy.ext.declarative import declarative_base


engine = create_engine('mysql+pymysql://root:laraiders@localhost:3306/stockmarket', echo=False)
db_session = scoped_session(sessionmaker(autocommit=False,
                                         autoflush=False,
                                         bind=engine))

def init_db():
    Model.metadata.create_all(bind=engine)


Model = declarative_base(name='Model')
Model.query = db_session.query_property()


class Stock(Model):

    __tablename__ = 'stock'

    id = Column(BigInteger, primary_key=True)
    symbol = Column(String(10), unique=True, nullable=False)
    name = Column(String(512), unique=False, nullable=False)
    enabled = Column(Boolean, nullable=True)
    advfn_url = Column(String(512), unique=False, nullable=True)
    sector = Column(String(512), unique=False, nullable=True)
    shares_in_issue = Column(BigInteger, unique=False, nullable=True)

    def __repr__(self):
        return '<Stock %r>' % self.name


class StockDaily(Model):

    __tablename__ = 'stock_daily'

    id = Column(BigInteger, primary_key=True)
    stock_id = Column(BigInteger, unique=False, nullable=False)
    symbol = Column(String(10), unique=False, nullable=False)
    price_date = Column(DateTime, unique=False, nullable=False)
    open_price = Column(Float, unique=False, nullable=True)
    close_price = Column(Float, unique=False, nullable=True)
    high_price = Column(Float, unique=False, nullable=True)
    low_price = Column(Float, unique=False, nullable=True)
    volume = Column(Float, unique=False, nullable=True)
