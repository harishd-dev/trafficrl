"""initial schema

Revision ID: 0001
Revises:
Create Date: 2024-01-01 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'training_sessions',
        sa.Column('id',             sa.String(36),  primary_key=True),
        sa.Column('status',         sa.String(20),  nullable=False, server_default='running'),
        sa.Column('algorithm',      sa.String(20),  nullable=False, server_default='DQN'),
        sa.Column('config',         sa.JSON(),       nullable=False),
        sa.Column('total_episodes', sa.Integer(),    nullable=False),
        sa.Column('best_reward',    sa.Float(),      nullable=False, server_default='0'),
        sa.Column('created_at',     sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('finished_at',    sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        'episode_logs',
        sa.Column('id',          sa.Integer(),  primary_key=True, autoincrement=True),
        sa.Column('session_id',  sa.String(36), sa.ForeignKey('training_sessions.id', ondelete='CASCADE'), nullable=False),
        sa.Column('episode',     sa.Integer(),  nullable=False),
        sa.Column('reward',      sa.Float(),    nullable=False),
        sa.Column('avg_wait',    sa.Float(),    nullable=False),
        sa.Column('throughput',  sa.Float(),    nullable=False),
        sa.Column('epsilon',     sa.Float(),    nullable=False),
        sa.Column('loss',        sa.Float(),    nullable=True),
        sa.Column('duration_s',  sa.Float(),    nullable=False),
    )
    op.create_index('ix_episode_logs_session_id', 'episode_logs', ['session_id'])
    op.create_index('ix_episode_logs_episode',    'episode_logs', ['episode'])


def downgrade() -> None:
    op.drop_table('episode_logs')
    op.drop_table('training_sessions')
